"""Opt-in testing controls. Native restoration arithmetic is reused verbatim.

Call finalize_mechanism_controls after every complete evaluation stream. Schedules
are trusted local torch artifacts, bound to source, anchors, Fisher and stream length.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap
import types

import torch
import sparse_restore


def binary_rga_ess(p, beta=0.3):
    if not math.isfinite(p) or not 0 <= p <= 1 or not 0 <= beta < 1:
        raise ValueError("binary disagreement p in [0,1] and beta in [0,1) required")
    return (1 - beta * p) ** 2 / (1 - (2 * beta - beta ** 2) * p)


def _digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _identity(r):
    return dict(source=hashlib.sha256(inspect.getsource(sparse_restore).encode()).hexdigest(),
                params=[(n, list(p.shape), _digest(r.anchors[n])) for n, p in r.params.items()],
                fisher=_digest(r._flat_fishers),
                momentum={n: getattr(r, n) for n in ("momentum_discounting", "momentum_away_factor",
                    "adam_exp_avg_decay", "adam_exp_avg_sq_decay", "adam_exp_avg_sq_decay_factor",
                    "online_fisher_beta", "online_fisher_alpha")})


def _trajectory_state(r):
    """Exact hashes: no tolerance silently hides changed adaptation or momentum."""
    def value(v):
        if torch.is_tensor(v):
            return (str(v.dtype), list(v.shape), _digest(v))
        if isinstance(v, dict):
            return {str(k): value(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, (list, tuple)):
            return [value(x) for x in v]
        return repr(v)
    return dict(parameters={n: value(p) for n, p in r.params.items()},
                optimizer=[[value(r.optimizer.state.get(p, {})) for p in group["params"]]
                           for group in r.optimizer.param_groups])


_POLICY_METADATA = ("last_base_selected_ratio", "last_selected_ratio", "last_requested_elements",
    "last_positive_score_elements", "last_fisher_rms", "last_parameter_drift_rms",
    "last_elastic_strength", "last_risk_score", "last_event_triggered", "num_scheduled_events",
    "num_skipped_low_reliability", "num_event_triggers", "event_consecutive_count", "cooldown_remaining")


class MechanismControls:
    def __init__(self, r, controller, beta, record, replay, ranking, seed):
        if record and replay:
            raise ValueError("record and replay are mutually exclusive")
        if controller not in {"native", "disagreement", "rga_ess"}:
            raise ValueError("unknown controller")
        if ranking not in {"fisher_drift", "drift_only", "fisher_only", "random"}:
            raise ValueError("unknown ranking")
        binary_rga_ess(0, beta)
        if r.num_batches:
            raise ValueError("install controls on a fresh restorer")
        self.r, self.controller, self.beta = r, controller, beta
        self.record, self.replay, self.ranking, self.seed = record, replay, ranking, seed
        self.reference_ranking = ranking if record else None
        self.identity = _identity(r)
        self.events, self.rows, self.audit = [], [], []
        self.calls = self.cursor = 0
        self.finished = False
        self.failed = False
        self.trajectory = []
        if replay:
            self.payload = torch.load(replay, map_location="cpu", weights_only=False)
            if self.payload["identity"] != self.identity:
                raise ValueError("schedule source/anchor/Fisher identity mismatch")
            self.reference_ranking = self.payload.get("reference_ranking", "fisher_drift")
            if self.reference_ranking not in {"fisher_drift", "drift_only"}:
                raise ValueError("unsupported reference ranking")
            if self.payload.get("controller", controller) != controller:
                raise ValueError("schedule controller mismatch")
            if r.selection_criterion != self.reference_ranking:
                raise ValueError("native ranking must match the recorded reference ranking")
            self.events = self.payload["events"]
            if len(self.payload.get("trajectory", [])) != self.payload["batches"]:
                raise ValueError("schedule lacks complete per-batch trajectory verification")
            if not self.events or not any(e["count"] for e in self.events):
                raise ValueError("zero-event schedule cannot support matched-budget comparison")
            previous = 0
            for e in self.events:
                if not previous < e["batch"] <= self.payload["batches"]:
                    raise ValueError("invalid schedule event ordering")
                previous = e["batch"]
                if (sum(e["quotas"]) != e["count"] or len(e["indices"]) != e["count"]
                    or len(e["quotas"]) != len(r.params)
                    or len(e["indices"].unique()) != e["count"]
                    or not torch.isin(e["indices"], e["eligible"]).all()):
                    raise ValueError("invalid schedule counts/quotas/eligibility")
        if record and (ranking not in {"fisher_drift", "drift_only"} or ranking != r.selection_criterion):
            raise ValueError("record ranking must match the native Fisher-drift or displacement policy")

    def _eligible(self, drift, event):
        r = self.r
        displacement = self.reference_ranking == "drift_only"
        f = None if displacement else r._get_hybrid_fishers()
        if r.hierarchical_budget and not event:
            masks = []
            for size, off in zip(r._parameter_sizes, r._global_offsets):
                b = drift[off:off+size]
                if displacement:
                    rms = b.square().mean().sqrt()
                    score = b / (rms + 1e-8) if rms > 1e-12 else torch.zeros_like(b)
                else:
                    a = f[off:off+size]
                    score = a * b if a.square().mean().sqrt() > 1e-12 and b.square().mean().sqrt() > 1e-12 else torch.zeros_like(a)
                masks.append(torch.isfinite(score) & (score > 0))
            return torch.cat(masks)
        if displacement:
            rms = drift.square().mean().sqrt()
            score = drift / (rms + 1e-8) if rms > 0 else torch.zeros_like(drift)
        else:
            score = f * drift if drift.square().mean().sqrt() > 0 else f.clone()
        if r.layer_aware and r.layer_weights:
            for (n, _), size, off in zip(r.params.items(), r._parameter_sizes, r._global_offsets):
                weight = next((float(w) for pat, w in r.layer_weights.items() if pat in n), 1.)
                score[off:off+size] *= weight
        return torch.isfinite(score) & (score > 0)

    def capture(self, indices, ratio, strength, event, clipped, drift):
        if not self.record:
            return
        eligible = self._eligible(drift, event) if ratio > 0 else torch.zeros_like(drift, dtype=torch.bool)
        if not eligible[indices].all():
            raise ValueError("reference selection outside finite-positive eligibility")
        quotas = [int(((indices >= off) & (indices < off+size)).sum())
                  for size, off in zip(self.r._parameter_sizes, self.r._global_offsets)]
        self.events.append(dict(batch=self.r.num_batches, ratio=ratio, strength=strength,
                                event=event, clipped=clipped, heavy=bool(ratio > .02 or event or clipped),
                                count=len(indices), dose=len(indices)*strength, quotas=quotas, indices=indices.clone(),
                                eligible=eligible.nonzero().flatten()))

    def replay_event(self, drift):
        r = self.r
        if self.cursor >= len(self.events) or self.events[self.cursor]["batch"] != r.num_batches:
            return None
        e = self.events[self.cursor]
        f = r._get_hybrid_fishers()
        generator = torch.Generator().manual_seed(self.seed + r.num_batches * 1000)
        score = {"fisher_drift": lambda: f * drift if drift.square().mean().sqrt() > 0 else f,
                 "drift_only": lambda: drift, "fisher_only": lambda: f,
                 "random": lambda: torch.rand(len(f), generator=generator)}[self.ranking]()
        chosen = []
        for size, off, quota in zip(r._parameter_sizes, r._global_offsets, e["quotas"]):
            candidates = e["eligible"][(e["eligible"] >= off) & (e["eligible"] < off+size)]
            if len(candidates) < quota:
                raise ValueError("reference quota exceeds eligible candidates")
            # Reference tie order guarantees native self replay even at exact ties.
            ref = e["indices"][(e["indices"] >= off) & (e["indices"] < off+size)]
            candidates = torch.cat((ref, candidates[~torch.isin(candidates, ref)]))
            values = score[candidates]
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite replay ranking score")
            chosen.append(candidates[torch.argsort(values, descending=True, stable=True)[:quota]])
        indices = torch.cat(chosen)
        if len(indices) != e["count"]:
            raise ValueError("replay count shortfall")
        if self.ranking == self.reference_ranking and not torch.equal(indices.sort().values, e["indices"].sort().values):
            raise ValueError("reference self replay coordinate mismatch (stream/trajectory differs)")
        self.cursor += 1
        self.audit.append(dict(batch=r.num_batches, count=len(indices), reference_count=e["count"],
                               strength=e["strength"], quotas=e["quotas"], agreement=True,
                               dose=len(indices)*e["strength"], heavy=e["heavy"], clipped=e["clipped"]))
        return e, indices

    def step(self, **kw):
        if self.failed or self.finished or self.r.num_batches != self.calls:
            raise ValueError("finished or reset stream: create fresh adapter and controls")
        if self.replay and self.calls >= self.payload["batches"]:
            raise ValueError("replay stream longer than reference")
        loss_ess = kw.get("ess_ratio")
        p = kw.get("anchor_disagreement")
        if self.controller != "native":
            if p is None:
                raise ValueError("controller requires preupdate binary anchor disagreement")
            weighted = binary_rga_ess(float(p), self.beta)
            kw["ess_ratio"] = 1. if self.controller == "disagreement" else weighted
        self.rows.append(dict(batch=self.calls+1, p=p, loss_ess=loss_ess, controller_ess=kw.get("ess_ratio")))
        try:
            check = self.record or (self.replay and self.ranking == self.reference_ranking)
            before = _trajectory_state(self.r) if check else None
            if self.replay and check and before != self.payload["trajectory"][self.calls]["before"]:
                raise ValueError(f"reference trajectory mismatch before batch {self.calls+1}")
            result = self.native(**kw)
            after = _trajectory_state(self.r) if check else None
            if self.record:
                self.trajectory.append(dict(before=before, after=after,
                    metadata={n:getattr(self.r,n) for n in _POLICY_METADATA if hasattr(self.r,n)}))
            if self.replay:
                expected = self.payload["trajectory"][self.calls]
                if check and after != expected["after"]:
                    raise ValueError(f"reference trajectory mismatch after batch {self.calls+1}")
                # These are reference schedule diagnostics, not candidate-derived policy decisions.
                for n,v in expected["metadata"].items():
                    setattr(self.r,n,v)
            self.calls += 1
            return result
        except Exception:
            self.failed = True
            raise

    def finalize(self, diagnostics_path=None):
        if self.failed:
            raise ValueError("invalidated controls after failed step: discard adapter and results")
        if self.record and not any(e["count"] for e in self.events):
            raise ValueError("zero recovered events: matched-budget experiment invalid")
        if self.replay and (self.cursor != len(self.events) or self.calls != self.payload["batches"]):
            raise ValueError("replay did not consume complete reference stream/events")
        if self.record:
            Path(self.record).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(identity=self.identity, events=self.events, batches=self.calls,
                            trajectory=self.trajectory, reference_ranking=self.reference_ranking,
                            controller=self.controller, trajectory_comparison="exact_sha256"), self.record)
        path = diagnostics_path or (str(self.record or self.replay)+f".{self.ranking}.audit.json" if self.record or self.replay else None)
        if path:
            schedule = self.record or self.replay
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(dict(controller=self.controller, beta=self.beta,
                ranking=self.ranking, reference_ranking=self.reference_ranking,
                selection_seed=self.seed, batches=self.calls,
                identity=self.identity,
                trajectory_verified=bool(self.replay and self.ranking == self.reference_ranking),
                trajectory_comparison="exact_sha256_per_batch_before_and_after_parameters_and_optimizer",
                policy_metadata="reference schedule values on replay",
                schedule_hash=hashlib.sha256(Path(schedule).read_bytes()).hexdigest() if schedule else None,
                eligibility=("fixed reference finite-positive displacement masks; reference tensor quotas"
                             if self.reference_ranking == "drift_only" else
                             "fixed reference finite-positive Fisher-drift masks; reference tensor quotas"),
                tie_policy="reference selection first for exactly tied scores", consumed_all=True,
                events=self.audit or [{k:v for k,v in e.items() if not torch.is_tensor(v)} for e in self.events],
                controller_diagnostics=self.rows), indent=2), encoding="utf-8")
        self.finished = True
        return path


def _instrument(r, controls):
    source = textwrap.dedent(inspect.getsource(type(r).step))
    source = source[source.index("def step("):]
    start = source.index("    # Evaluate Triple-Condition")
    end = source.index("    self.selected = self._global_indices_to_selection(global_indices)", start)
    native_policy = source[start:end]
    replacement = "    if self._revision_controls.replay:\n        replay = self._revision_controls.replay_event(self._current_absolute_drift())\n        if replay is None:\n            return 0\n        e, global_indices = replay\n        selected_ratio, strength = e['ratio'], e['strength']\n        event_matched, gradient_clipped = e['event'], e['clipped']\n        base_ratio = selected_ratio\n        absolute_drift = self._current_absolute_drift()\n        self.num_scheduled_events += 1\n        self.last_selected_ratio = selected_ratio\n        self.last_elastic_strength = strength\n        self.last_event_triggered = event_matched\n    else:\n"
    replacement += textwrap.indent(native_policy, "    ")
    replacement += "    self._revision_controls.capture(global_indices, selected_ratio, strength, event_matched, gradient_clipped, absolute_drift)\n"
    source = source[:start] + replacement + source[end:]
    namespace = dict(vars(sparse_restore))
    exec(compile(source, "<revision-native-restorer-step>", "exec"), namespace)
    r._revision_controls = controls
    controls.native = types.MethodType(torch.no_grad()(namespace["step"]), r)
    r.step = controls.step


def install_mechanism_controls(adapter, *, controller="native", beta=0.3,
                               schedule_record=None, schedule_replay=None,
                               ranking="fisher_drift", selection_seed=20261018):
    if controller != "native" and getattr(adapter, "rga_ablation_variant", None) not in {"uniform_recap", "anchor_only"}:
        raise ValueError("factorial controller requires binary uniform_recap or anchor_only loss")
    if controller != "native" and adapter.rga_ablation_variant == "anchor_only" and abs(getattr(adapter, "anchor_rectify_beta", -1)-beta) > 1e-12:
        raise ValueError("loss anchor_rectify_beta must match controller beta")
    r = adapter.restorer
    if type(r) is not sparse_restore.PeriodicSparseRestorer:
        raise TypeError("controls require exact native PeriodicSparseRestorer")
    if not schedule_record and not schedule_replay:
        r.selection_criterion, r.selection_seed = ranking, selection_seed
    controls = MechanismControls(r, controller, beta, schedule_record, schedule_replay, ranking, selection_seed)
    _instrument(r, controls)
    adapter.mechanism_controls = controls
    return adapter


def finalize_mechanism_controls(adapter, diagnostics_path=None):
    return adapter.mechanism_controls.finalize(diagnostics_path)
