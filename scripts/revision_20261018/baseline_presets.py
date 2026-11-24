"""Opt-in, fixed author-informed presets; no target-label model selection.

Equations are independently implemented from the author repositories:
https://github.com/mr-eggplant/EATA/blob/main/eata.py
https://github.com/hzcar/ReCAP/blob/main/recap.py
https://github.com/YonseiML/asr/blob/main/conf.py
No third-party source is vendored. ASR uses the author's ETA backend with online
Fisher/mean consolidation, with an explicit finite empty-selection safeguard.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F

METHODS = {'eata', 'asr', 'sar', 'recap'}


def configure_reasonable(args):
    if args.tta_method not in METHODS:
        return args
    args.batch_size = 128
    args.optimizer = 'sgd'
    args.adapt_lr = .00025
    args.momentum = .9
    args.weight_decay = 0.
    args.sar_sam_rho = .05
    args.sar_margin_factor = .4
    args.sar_reset_threshold = .2
    args.eata_e_margin_factor = .4
    args.eata_d_margin = .05
    args.eata_fisher_alpha = 2000.
    args.asr_anchor_fisher_alpha = 0.
    args.asr_anchor_l2_alpha = 0.
    args.adapt_bn_scope = 'method_default'
    args.fslr_grad_clip_norm = 0.
    args.recap_confidence_gate = 'none'
    return args


@contextmanager
def reasonable_factory_scope(args):
    """Single-threaded factory transaction; restore even on validation failure."""
    import evaluator.engine as engine
    original = engine.configure_bn_adaptation
    if args.tta_method == 'eata':
        engine.configure_bn_adaptation = engine.configure_official_bn_adaptation
    try:
        yield
    finally:
        engine.configure_bn_adaptation = original


def eata_objective(entropy, selected, margin):
    chosen = entropy[selected]
    return (chosen * torch.exp(float(margin) - chosen.detach())).mean()


def recap_objective(proxy, logits):
    regional_entropy, regional_instability = proxy(logits)
    return (regional_entropy + .5 * regional_instability).mean()


class ASROnlineMemory:
    """Online pre-update gradient second moments and parameter means.

    Segment averages and EMA orientation follow YonseiML/asr methods/base.py.
    These estimates require no training labels or additional source examples.
    """
    def __init__(self, params):
        self.params=list(params)
        self.fishers=None
        self.means=None
        self.buffer_count=0
        self.fisher_sums=[torch.zeros_like(p) for p in self.params]
        self.mean_sums=[torch.zeros_like(p) for p in self.params]

    def observe(self,loss):
        gradients=torch.autograd.grad(loss,self.params,retain_graph=True)
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            return False
        with torch.no_grad():
            for f,m,p,g in zip(self.fisher_sums,self.mean_sums,self.params,gradients):
                f.add_(g.detach().square())
                m.add_(p.detach())
        self.buffer_count+=1
        return True

    def consolidate(self,steps):
        if steps != self.buffer_count or steps<=0:
            raise ValueError('ASR segment must contain exactly one observation per batch')
        fisher=[f/steps for f in self.fisher_sums]
        means=[m/steps for m in self.mean_sums]
        self.fishers=fisher if self.fishers is None else [.9*old+.1*new for old,new in zip(self.fishers,fisher)]
        self.means=means if self.means is None else [.9*old+.1*new for old,new in zip(self.means,means)]
        for value in self.fisher_sums+self.mean_sums:
            value.zero_()
        self.buffer_count=0

    def penalty(self,coefficient):
        if self.fishers is None:
            return self.params[0].sum()*0.
        return float(coefficient)*sum((f*(p-m).square()).sum() for p,f,m in zip(self.params,self.fishers,self.means))


@torch.enable_grad()
def _asr_adapt(self,images):
    self.optimizer.zero_grad(set_to_none=True)
    logits=self.model(images)
    logp=F.log_softmax(logits,1)
    probabilities=logp.exp()
    entropy=-(probabilities*logp).sum(1)
    selected=entropy.detach()<self.eata_e_margin
    if self.current_model_probs is not None:
        selected &= F.cosine_similarity(self.current_model_probs.unsqueeze(0),probabilities.detach(),dim=1).abs()<self.eata_d_margin
    self._update_model_probs(probabilities.detach()[selected])
    # Upstream empty mean is NaN; this explicit zero-gradient observation
    # preserves the batch clock without poisoning subsequent online memory.
    base_loss=eata_objective(entropy,selected,self.eata_e_margin) if selected.any() else entropy.sum()*0.
    if not torch.isfinite(base_loss):
        raise FloatingPointError('nonfinite ASR objective')
    self.reasonable_nonfinite_gradient=False
    self.reasonable_gradient_norm=None
    if not self.reasonable_memory.observe(base_loss):
        self.reasonable_nonfinite_gradient=True
        return _record(self,images,logits,entropy,selected,base_loss,False)
    consolidation=self.reasonable_memory.penalty(self.reasonable_previous_p_fisher)
    loss=base_loss+consolidation
    updated=bool(selected.any()) and bool(torch.isfinite(loss))
    if updated:
        loss.backward()
        updated=_finite_gradient_step(self)
    self.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        average_logits=logits.detach().mean(0,keepdim=True)
        self.asr_entropy=float(-(average_logits.softmax(1)*average_logits.log_softmax(1)).sum())
        source_prediction=self.source_model(images).argmax(1)
        flip=float(source_prediction.ne(logits.detach().argmax(1)).float().mean())
    self.asr_step+=1
    self.asr_last_flip_rate=flip
    self.asr_last_fisher_scale=self.asr_p0*flip**2
    self.reasonable_previous_p_fisher=self.asr_last_fisher_scale
    reset=self.asr_entropy<self.asr_cum_entropy
    if reset:
        self.reasonable_memory.consolidate(self.asr_step)
        self._reset_from_anchor()
        self.current_model_probs=None
    else:
        momentum=self.asr_m0*flip+1-self.asr_m0
        self.asr_cum_entropy=momentum*self.asr_cum_entropy+(1-momentum)*self.asr_entropy
    outputs,stats=_record(self,images,logits,entropy,selected,loss,updated,reset)
    stats.update(asr_reset=reset,asr_scope=self.asr_last_scope,asr_flip_rate=flip,
                 asr_fisher_scale=self.asr_last_fisher_scale,asr_entropy=self.asr_entropy,
                 asr_consolidation_loss=float(consolidation.detach()))
    self.counters['asr_entropy_sum']+=self.asr_entropy
    self.counters['asr_flip_rate_sum']+=flip
    self.counters['asr_fisher_scale_sum']+=self.asr_last_fisher_scale
    return outputs,stats


def _finite_gradient_step(adapter):
    gradients=[p.grad for p in adapter.model.parameters() if p.requires_grad and p.grad is not None]
    finite=all(bool(torch.isfinite(g).all()) for g in gradients)
    norm=torch.sqrt(sum(g.detach().double().square().sum() for g in gradients)) if gradients else torch.tensor(0.)
    finite=finite and bool(torch.isfinite(norm))
    adapter.reasonable_nonfinite_gradient=not finite
    adapter.reasonable_gradient_norm=float(norm) if finite else None
    if finite:
        adapter.optimizer.step()
    return finite


def _record(adapter, images, logits, entropy, selected, loss, updated, reset=False):
    n = len(images)
    count = int(selected.sum())
    values = dict(samples=n, quality_passed=count, margin_passed=n,
                  margin_sum=0., entropy_passed=count, nonredundant=count,
                  region_reliable=n, selected=count, updated=updated,
                  nonfinite=not bool(torch.isfinite(loss)) or getattr(adapter,'reasonable_nonfinite_gradient',False), loss=float(loss.detach()) if torch.isfinite(loss) else 0.,
                  prediction_entropy=float(entropy.mean().detach()),
                  recap_loss=float(loss.detach()) if adapter.tta_method=='recap' and torch.isfinite(loss) else 0.,
                  eata_regularization=0., recovered=adapter.restorer.step(),
                  gradient_norm_before_clip=getattr(adapter,'reasonable_gradient_norm',None), gradient_clipped=False,
                  nonfinite_gradient=getattr(adapter,'reasonable_nonfinite_gradient',False), source_agreement=None,
                  reasonable_reset=reset)
    for key in ('samples','quality_passed','margin_passed','margin_sum','entropy_passed','nonredundant','region_reliable','selected'):
        adapter.counters[key] += values[key]
    adapter.counters['batches'] += 1
    adapter.counters['updated_batches'] += int(updated)
    adapter.counters['skipped_batches'] += int(not updated)
    adapter.counters['nonfinite_batches'] += int(values['nonfinite'])
    adapter.counters['nonfinite_gradient_batches'] += int(values['nonfinite_gradient'])
    if values['gradient_norm_before_clip'] is not None:
        norm=values['gradient_norm_before_clip']
        adapter.counters['gradient_norm_batches']+=1
        adapter.counters['gradient_norm_sum']+=norm
        adapter.counters['gradient_norm_max']=max(adapter.counters['gradient_norm_max'],norm)
    adapter.counters['loss_sum'] += values['loss']
    adapter.counters['recap_loss_sum'] += values['recap_loss']
    return logits.detach(), values


@torch.enable_grad()
def _eata_adapt(self, images):
    self.reasonable_nonfinite_gradient=False
    self.reasonable_gradient_norm=None
    self.optimizer.zero_grad(set_to_none=True)
    logits = self.model(images)
    logp = F.log_softmax(logits,1)
    probability = logp.exp()
    entropy = -(probability*logp).sum(1)
    reliable = entropy.detach() < self.eata_e_margin
    selected = reliable.clone()
    if self.current_model_probs is not None:
        similarity = F.cosine_similarity(self.current_model_probs.unsqueeze(0),probability.detach(),dim=1).abs()
        selected &= similarity < self.eata_d_margin
    self._update_model_probs(probability.detach()[selected])
    regularizer = self._eata_regularization()
    loss = entropy.sum()*0.
    updated = False
    if selected.any():
        loss = eata_objective(entropy,selected,self.eata_e_margin)
        if regularizer is not None:
            loss = loss + regularizer
        if torch.isfinite(loss):
            loss.backward()
            updated = _finite_gradient_step(self)
    self.optimizer.zero_grad(set_to_none=True)
    output, stats = _record(self,images,logits,entropy,selected,loss,updated)
    stats['entropy_passed'] = int(reliable.sum())
    self.counters['entropy_passed'] += int(reliable.sum())-int(selected.sum())
    stats['eata_regularization'] = float(regularizer.detach()) if regularizer is not None else 0.
    self.counters['eata_regularization_sum'] += stats['eata_regularization']
    return output, stats


@torch.enable_grad()
def _recap_adapt(self, images):
    self.reasonable_nonfinite_gradient=False
    self.reasonable_gradient_norm=None
    self.optimizer.zero_grad(set_to_none=True)
    logits = self.model(images)
    entropy = -(logits.softmax(1)*logits.log_softmax(1)).sum(1)
    loss = recap_objective(self.proxy,logits)
    updated = bool(torch.isfinite(loss))
    reset = False
    if updated:
        value = float(loss.detach())/2.
        self.reasonable_ema = value if self.reasonable_ema is None else .9*self.reasonable_ema+.1*value
        loss.backward()
        updated=_finite_gradient_step(self)
        if updated and self.reasonable_ema < .2:
            self.model.load_state_dict(self.reasonable_source_state)
            self.optimizer.load_state_dict(deepcopy(self.reasonable_optimizer_state))
            reset = True
    self.optimizer.zero_grad(set_to_none=True)
    return _record(self,images,logits,entropy,torch.ones_like(entropy,dtype=torch.bool),loss,updated,reset)


@torch.no_grad()
def _sam_rollback(optimizer):
    """Undo only the SAM perturbation, without touching SGD momentum."""
    for group in optimizer.param_groups:
        for parameter in group['params']:
            old=optimizer.state[parameter].pop('old_p',None)
            if old is not None:
                parameter.copy_(old)


def _sar_gradients(adapter):
    gradients=[p.grad for p in adapter.model.parameters() if p.grad is not None]
    finite=bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    norm=torch.sqrt(sum(g.detach().double().square().sum() for g in gradients)) if gradients else torch.tensor(0.)
    finite=finite and bool(torch.isfinite(norm))
    adapter.reasonable_nonfinite_gradient=not finite
    adapter.reasonable_gradient_norm=float(norm) if finite else None
    return finite


@torch.enable_grad()
def _sar_adapt(self,images):
    self.optimizer.zero_grad(set_to_none=True)
    self.reasonable_nonfinite_gradient=False
    self.reasonable_gradient_norm=None
    outputs=self.model(images)
    entropy=-(outputs.softmax(1)*outputs.log_softmax(1)).sum(1)
    first=(entropy.detach()<self.margin).nonzero(as_tuple=False).flatten()
    second_count=0
    updated=False
    reset=False
    loss_value=0.
    finite=bool(torch.isfinite(entropy).all())
    if len(first) and finite:
        first_loss=entropy[first].mean()
        first_loss.backward()
        if _sar_gradients(self):
            self.optimizer.first_step(zero_grad=True)
            try:
                perturbed=self.model(images)
                second_entropy=-(perturbed.softmax(1)*perturbed.log_softmax(1)).sum(1)[first]
                finite=bool(torch.isfinite(second_entropy).all())
                second=second_entropy.detach()<self.margin
                second_count=int(second.sum())
                if finite and second_count:
                    loss=second_entropy[second].mean()
                    loss_value=float(loss.detach())
                    loss.backward()
                    if _sar_gradients(self):
                        # Restore all perturbed tensors even if this pass has
                        # no gradient for a parameter that had one previously.
                        _sam_rollback(self.optimizer)
                        self.optimizer.base_optimizer.step()
                        updated=True
                        self.ema=loss_value if self.ema is None else .9*self.ema+.1*loss_value
                        if self.ema<self.reset_threshold:
                            self.model.load_state_dict(self.source_state)
                            self.optimizer.load_state_dict(deepcopy(self.source_optimizer_state))
                            self.optimizer.base_optimizer.load_state_dict(deepcopy(self.reasonable_base_optimizer_state))
                            reset=True
            finally:
                _sam_rollback(self.optimizer)
    self.optimizer.zero_grad(set_to_none=True)
    finite=finite and not self.reasonable_nonfinite_gradient
    stats=self._record(len(images),len(first),second_count,loss_value if finite else float('nan'),updated,
        sar_first_selected=len(first),sar_second_selected=second_count,sar_ema=self.ema,sar_reset=reset,
        recovered=self.restorer.step(),nonfinite_gradient=self.reasonable_nonfinite_gradient,
        gradient_norm_before_clip=self.reasonable_gradient_norm,gradient_clipped=False)
    self.counters['nonfinite_gradient_batches']+=int(self.reasonable_nonfinite_gradient)
    if self.reasonable_gradient_norm is not None:
        norm=self.reasonable_gradient_norm
        self.counters['gradient_norm_batches']+=1
        self.counters['gradient_norm_sum']+=norm
        self.counters['gradient_norm_max']=max(self.counters.get('gradient_norm_max',0.),norm)
    return outputs.detach(),stats


def install_reasonable_adapter(adapter, metadata, args):
    method = args.tta_method
    if method not in METHODS:
        return adapter, metadata
    if method == 'eata':
        # A missing factory context must fail before any adaptation occurs.
        expected = {n for n,p in adapter.model.named_parameters() if p.requires_grad}
        all_bn = {f'{n}.{k}' for n,m in adapter.model.named_modules() if isinstance(m,torch.nn.BatchNorm2d) for k,p in m.named_parameters(recurse=False) if k in ('weight','bias')}
        if expected != all_bn:
            raise RuntimeError('EATA requires reasonable_factory_scope around original_factory')
        adapter.adapt = MethodType(_eata_adapt,adapter)
        adapter.sample_filter = 'eata_entropy_diversity_author_reweight'
    elif method == 'recap':
        adapter.adapt = MethodType(_recap_adapt,adapter)
        adapter.reasonable_ema = None
        adapter.reasonable_source_state = deepcopy(adapter.model.state_dict())
        adapter.reasonable_optimizer_state = deepcopy(adapter.optimizer.state_dict())
        adapter.sample_filter = 'all_samples_basic_recap'
    elif method == 'asr':
        adapter.optimizer=torch.optim.SGD([p for p in adapter.model.parameters() if p.requires_grad],lr=.00025,momentum=.9,nesterov=True,weight_decay=0.)
        adapter.initial_optimizer_state=deepcopy(adapter.optimizer.state_dict())
        adapter.adapt=MethodType(_asr_adapt,adapter)
        adapter.reasonable_memory=ASROnlineMemory([p for p in adapter.model.parameters() if p.requires_grad])
        adapter.reasonable_previous_p_fisher=0.
        adapter.sample_filter='eta_entropy_diversity_author_reweight'
    elif method == 'sar':
        if adapter.steps!=1:
            raise ValueError('reasonable SAR preset requires one complete SAM update')
        adapter.adapt=MethodType(_sar_adapt,adapter)
        adapter.reasonable_base_optimizer_state=deepcopy(adapter.optimizer.base_optimizer.state_dict())
        adapter.sample_filter='sar_two_stage_entropy'
        adapter.tta_method='sar'
    metadata = dict(metadata)
    metadata['reasonable_preset'] = f'{method}_author_informed_fedacg' if method!='asr' else 'eta_asr_online_consolidation_fedacg'
    metadata['official_replication'] = False
    metadata['reasonable_implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['reasonable_upstream_sources']={
        'https://raw.githubusercontent.com/mr-eggplant/EATA/main/eata.py':'f162467d300cb12a14a3c888f693b31d15b12cd116ba0edadfa6560fa8b6ef96',
        'https://raw.githubusercontent.com/mr-eggplant/SAR/main/sar.py':'0553a395ac2bc087049720f1f162789079fe5ee25aa6a1cf82b2ea6286d66eff',
        'https://raw.githubusercontent.com/hzcar/ReCAP/main/recap.py':'4e4f0a94e6587d010b22101444748327f04fd879d4c50cb6b71f166ed025ab46',
        'https://raw.githubusercontent.com/YonseiML/asr/main/methods/eata.py':'96584da6a8153179153acba0737bd0372184a4210689d7c017882759d099de83',
        'https://raw.githubusercontent.com/YonseiML/asr/main/methods/base.py':'aaf2d2edf869a481f5db7f29e7ce9f96d151a9a6874aaa2cce13062a2962553b',
        'https://raw.githubusercontent.com/YonseiML/asr/main/cfgs/cin_c/eta/eta0.yaml':'d485db77ba0319c79ffb7bed9fa1c6abe5ee7a395e02d9e3cb9805b581056966',
    }
    metadata['reasonable_limitations'] = ['FedACG backbone and batch128 transfer']
    if method=='eata':
        metadata['reasonable_source_fisher_protocol']={'cache':'eata_official','target':'pseudo_label','normalization':'target_batch','samples':2000,'batches':16,'scope':'all_bn','statistics_path':getattr(args,'statistics_path',None)}
    if method=='asr':
        metadata['reasonable_limitations'].append('empty ETA selection uses zero-gradient online observation instead of upstream NaN mean')
    metadata['optimizer'] = 'adam' if isinstance(adapter.optimizer,torch.optim.Adam) else ('sam_sgd' if method=='sar' else 'sgd')
    names=[n for n,p in adapter.model.named_parameters() if p.requires_grad]
    metadata['trainable_parameter_names']=names
    metadata['trainable_parameter_tensors']=len(names)
    metadata['trainable_parameter_scalars']=sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    metadata['parameter_scope']='bn_affine_all' if method in ('asr','eata') else 'norm_affine_excluding_layer4'
    metadata['sample_filter']=adapter.sample_filter
    metadata['reasonable_actual_optimizer_groups']=[{k:v for k,v in g.items() if k!='params'} for g in adapter.optimizer.param_groups]
    return adapter,metadata
