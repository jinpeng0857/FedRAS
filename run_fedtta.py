"""
FedTTA Main Entry Point: Domain-Isolated Federated Test-Time Adaptation.

Modes:
  - source_only:          No TTA, direct evaluation
  - local_tent:           Each client runs Tent independently
  - local_eata:           Each client runs EATA independently
  - local_eata_c:         Each client runs EATA-C independently
  - naive_fedavg_tta:     Single shared adapter, FedAvg (all domains mixed)
  - naive_fedacg_tta:     Single shared adapter+momentum, FedACG (all domains mixed)
  - domain_fedavg_eatac:  Per-domain adapter, FedAvg within domain
  - domain_fedacg_eatac:  Per-domain adapter+momentum, FedACG within domain ★ MAIN

Key principles:
  - Different corruptions/domains NEVER share adapter or momentum.
  - Each client runs local EATA-C in complete isolation per round.
  - After each round, clients reset to source model + server adapter.
"""

import copy
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import (
    STANDARD_CORRUPTIONS, DATASET_CONFIGS, load_clean_test,
)
from src.model.eatac_resnet import eatac_resnet18, eatac_resnet34
from src.model.eatac_imagenet_resnet import eatac_resnet50, eatac_resnet101
from src.algorithm.EATA_C.losses import prediction_agreement, min_max_entropy_regularization
from src.algorithm.EATA_C.sample_selection import ActiveSampleSelector
from src.algorithm.FedTTA import (
    FedTTAClient,
    FedTTAServer,
    extract_adapt_state,
    weighted_average_delta,
    zeros_like_state,
    apply_state_delta,
    add_state,
    scale_state,
    compute_accuracy,
    compute_ece,
    compute_client_reliability,
)
from dataset.target_clients import build_single_corruption_target_clients
from utils.config import parser
from utils.logger import logger, add_file_handler
from utils.utils import set_seed, load_checkpoint_weights


# ========== Baseline Implementations ==========


def run_source_only(target_clients, model, device):
    """
    Baseline: No adaptation. Evaluate all target clients via TWO protocols.

    1. source_only_running_stats: model.eval(), BN uses running stats (traditional).
    2. source_only_batchstats:   BN.train() + save/restore, BN uses batch stats.
       This is the correct baseline for comparing adapt_lr=0 / adapt_steps=0.
    """
    logger.info("=" * 60)
    logger.info("Baseline: Source-Only (two BN protocols)")
    logger.info("=" * 60)

    source_model = copy.deepcopy(model).to(device)

    # ─── Helper: batch-stats forward on a single client's data ───
    def _batchstats_eval(bs_model, loader):
        bn_saved = {}
        for nm, md in bs_model.named_modules():
            if isinstance(md, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_saved[nm] = (md.running_mean.data.clone(), md.running_var.data.clone())
        bs_model.eval()
        for md in bs_model.modules():
            if isinstance(md, (nn.BatchNorm1d, nn.BatchNorm2d)):
                md.train()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = bs_model(x)
                log = out["logit"] if isinstance(out, dict) else out
                logits_list.append(log.cpu())
                labels_list.append(y.cpu())
        for nm, md in bs_model.named_modules():
            if isinstance(md, (nn.BatchNorm1d, nn.BatchNorm2d)) and nm in bn_saved:
                md.running_mean.data.copy_(bn_saved[nm][0])
                md.running_var.data.copy_(bn_saved[nm][1])
        all_log = torch.cat(logits_list, dim=0)
        all_lbl = torch.cat(labels_list, dim=0)
        probs = F.softmax(all_log, dim=1)
        return compute_accuracy(all_log, all_lbl), compute_ece(probs, all_lbl)

    # ─── Protocol A: BN running stats (traditional model.eval()) ───
    logger.info("--- Source-Only (running-stats) ---")
    rs_per_client, rs_domain = [], defaultdict(list)
    rs_model = copy.deepcopy(source_model).to(device)
    rs_model.eval()

    for client in target_clients:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in client["test_loader"]:
                x, y = x.to(device), y.to(device)
                out = rs_model(x)
                logits = out["logit"] if isinstance(out, dict) else out
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        probs = F.softmax(all_logits, dim=1)
        acc = compute_accuracy(all_logits, all_labels)
        ece_val = compute_ece(probs, all_labels)
        rs_per_client.append((client["client_id"], client["domain_id"], acc, ece_val))
        rs_domain[client["domain_id"]].append((acc, ece_val))

    for d_id, vals in sorted(rs_domain.items()):
        d_acc = np.mean([v[0] for v in vals])
        d_ece = np.mean([v[1] for v in vals])
        corr = target_clients[[c["domain_id"] for c in target_clients].index(d_id)]["corruption"]
        logger.info(f"  Domain {d_id} ({corr}): Acc={d_acc:.2%} ECE={d_ece:.2%}")
    rs_avg_acc = np.mean([v[2] for v in rs_per_client])
    rs_avg_ece = np.mean([v[3] for v in rs_per_client])
    logger.info(f"  => Source-only (running-stats) Avg Acc={rs_avg_acc:.2%} Avg ECE={rs_avg_ece:.2%}")

    # ─── Protocol B: BN batch statistics ───
    logger.info("--- Source-Only (batch-stats) ---")
    bs_per_client, bs_domain = [], defaultdict(list)

    for client in target_clients:
        acc, ece_val = _batchstats_eval(copy.deepcopy(source_model), client["test_loader"])
        bs_per_client.append((client["client_id"], client["domain_id"], acc, ece_val))
        bs_domain[client["domain_id"]].append((acc, ece_val))

    for d_id, vals in sorted(bs_domain.items()):
        d_acc = np.mean([v[0] for v in vals])
        d_ece = np.mean([v[1] for v in vals])
        corr = target_clients[[c["domain_id"] for c in target_clients].index(d_id)]["corruption"]
        logger.info(f"  Domain {d_id} ({corr}): Acc={d_acc:.2%} ECE={d_ece:.2%}")
    bs_avg_acc = np.mean([v[2] for v in bs_per_client])
    bs_avg_ece = np.mean([v[3] for v in bs_per_client])
    logger.info(f"  => Source-only (batch-stats) Avg Acc={bs_avg_acc:.2%} Avg ECE={bs_avg_ece:.2%}")

    # ─── Summary ───
    logger.info("=" * 60)
    logger.info(f"  Source-only running-stats: Acc={rs_avg_acc:.2%} ECE={rs_avg_ece:.2%}")
    logger.info(f"  Source-only batch-stats:   Acc={bs_avg_acc:.2%} ECE={bs_avg_ece:.2%}")
    logger.info(f"  NOTE: adapt_lr=0 / adapt_steps=0 should match batch-stats baseline.")
    logger.info("=" * 60)

    return {
        "avg_acc": rs_avg_acc,
        "avg_ece": rs_avg_ece,
        "batch_stats_acc": bs_avg_acc,
        "batch_stats_ece": bs_avg_ece,
    }


def run_local_tent(target_clients, base_model, device, configs):
    """Baseline: Each client runs Tent independently (no communication)."""
    logger.info("=" * 60)
    logger.info("Baseline: Local Tent (no communication)")
    logger.info("=" * 60)

    all_accs = []
    for client in target_clients:
        model = copy.deepcopy(base_model).to(device)
        # Respect freeze_bn: BN.eval() keeps running stats, affine still trainable
        if getattr(configs, 'freeze_bn', True):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True
                else:
                    m.requires_grad_(False)
        else:
            model.train()
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.requires_grad_(True)
                    for p in m.parameters():
                        p.requires_grad = True
                else:
                    m.requires_grad_(False)

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=configs.adapt_lr, momentum=0.9,
        )

        all_logits, all_labels = [], []
        for x, y in client["test_loader"]:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out["logit"] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()
            with torch.no_grad():
                all_logits.append(logits.detach().cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        acc = compute_accuracy(all_logits, torch.cat(all_labels))
        all_accs.append(acc)

    avg_acc = np.mean(all_accs)
    logger.info(f"  Local Tent Avg Acc: {avg_acc:.2%}")
    return {"avg_acc": avg_acc}


def run_local_eata(target_clients, base_model, device, configs):
    """Baseline: Each client runs EATA independently."""
    logger.info("=" * 60)
    logger.info("Baseline: Local EATA (no communication)")
    logger.info("=" * 60)

    e0 = configs.e_margin * np.log(configs.num_class)
    all_accs = []

    for client in target_clients:
        model = copy.deepcopy(base_model).to(device)
        if getattr(configs, 'freeze_bn', True):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True
                elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    for p in m.parameters():
                        p.requires_grad = True
                else:
                    m.requires_grad_(False)
        else:
            model.train()
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for p in m.parameters():
                        p.requires_grad = True
                else:
                    m.requires_grad_(False)

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=configs.adapt_lr, momentum=0.9,
        )
        selector = ActiveSampleSelector(e0=e0, epsilon=configs.epsilon, alpha_ma=configs.alpha_ma)

        all_logits, all_labels = [], []
        is_first = True

        for x, y in client["test_loader"]:
            x, y = x.to(device), y.to(device)

            # Forward for selection (no grad)
            with torch.no_grad():
                out = model(x)
                full_logits = out["logit"] if isinstance(out, dict) else out

            scores, selected = selector.select(full_logits, is_first_batch=is_first)
            is_first = False

            if len(selected) > 0:
                for _ in range(configs.adapt_steps):
                    # Re-enable grad for selected samples
                    if configs.adapt_steps == 1:
                        # Single step: forward on selected with grad
                        selected_out = model(x)  # forward full batch for BN stats
                        selected_logits = (selected_out["logit"] if isinstance(selected_out, dict) else selected_out)[selected]
                    else:
                        selected_out = model(x)
                        selected_logits = (selected_out["logit"] if isinstance(selected_out, dict) else selected_out)[selected]

                    selected_probs = F.softmax(selected_logits, dim=1)
                    loss = -(selected_probs * (selected_probs + 1e-8).log()).sum(dim=1).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            all_logits.append(full_logits.detach().cpu())
            all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        acc = compute_accuracy(all_logits, torch.cat(all_labels))
        all_accs.append(acc)

    avg_acc = np.mean(all_accs)
    logger.info(f"  Local EATA Avg Acc: {avg_acc:.2%}")
    return {"avg_acc": avg_acc}


def run_local_eata_c(target_clients, base_model, device, configs, fisher_values=None):
    """Baseline: Each client runs full EATA-C independently."""
    logger.info("=" * 60)
    logger.info("Baseline: Local EATA-C (no communication)")
    logger.info("=" * 60)

    from src.algorithm.EATA_C.eata_c_test import EATACTester
    from src.algorithm.FedTTA import extract_adapt_state, load_adapt_state, compute_accuracy, compute_ece

    online_accs, online_eces = [], []
    adapter_accs, adapter_eces = [], []
    for client in target_clients:
        model = copy.deepcopy(base_model)
        tester = EATACTester(model, configs, device, fisher_path=None)
        if fisher_values is not None:
            tester.fisher_values = fisher_values

        # === TTA phase (online accuracy) ===
        result = tester.evaluate(client["test_loader"])
        online_accs.append(result["accuracy"])
        online_eces.append(result["ece"])

        # === Adapter evaluation phase (pure forward, no TTA) ===
        # Save adapted state, reset to source, load adapted state, pure forward
        adapted_state = extract_adapt_state(tester.model)
        tester._reset_model()
        load_adapt_state(tester.model, adapted_state)
        tester.model.eval()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in client["test_loader"]:
                x, y = x.to(device), y.to(device)
                out = tester.model(x)
                logits = out["logit"] if isinstance(out, dict) else out
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        adapter_acc = compute_accuracy(all_logits, all_labels)
        adapter_ece = compute_ece(F.softmax(all_logits, dim=1), all_labels)
        adapter_accs.append(adapter_acc)
        adapter_eces.append(adapter_ece)

    avg_online_acc = np.mean(online_accs)
    avg_online_ece = np.mean(online_eces)
    avg_adapter_acc = np.mean(adapter_accs)
    avg_adapter_ece = np.mean(adapter_eces)
    logger.info(f"  Local EATA-C Online  Acc={avg_online_acc:.2%} ECE={avg_online_ece:.2%}")
    logger.info(f"  Local EATA-C Adapter Acc={avg_adapter_acc:.2%} ECE={avg_adapter_ece:.2%}")
    return {"avg_acc": avg_adapter_acc, "avg_ece": avg_adapter_ece,
            "online_acc": avg_online_acc, "online_ece": avg_online_ece}


# ========== Naive FedTTA Baselines (single shared adapter, no domain isolation) ==========


def run_naive_fedavg_tta(target_clients, base_model, device, configs, fisher_values):
    """
    Baseline: Single shared adapter, FedAvg aggregation (all domains mixed).

    Used to prove that mixing different corruptions causes negative transfer.
    """
    logger.info("=" * 60)
    logger.info("Baseline: Naive FedAvg-TTA (no domain awareness)")
    logger.info("=" * 60)

    # Create client objects
    client_objects = {}
    for info in target_clients:
        client_objects[info["client_id"]] = FedTTAClient(
            client_id=info["client_id"], domain_id=info["domain_id"],
            corruption=info["corruption"], base_model=base_model,
            device=device, configs=configs, fisher_values=fisher_values,
        )

    source_state = extract_adapt_state(base_model)
    shared_adapter = copy.deepcopy(source_state)
    tta_rounds = configs.tta_rounds
    join_ratio = getattr(configs, "target_participation_rate", 1.0)

    round_metrics_list = []
    is_first = True

    for round_idx in range(tta_rounds):
        round_start = time.time()

        # Sample clients
        num_to_sample = max(1, int(len(client_objects) * join_ratio))
        selected_cids = np.random.choice(
            list(client_objects.keys()), num_to_sample, replace=False
        ).tolist()

        # Collect client results (each client runs local EATA-C once per round)
        deltas, weights = [], []
        per_result_accs = []

        for cid in selected_cids:
            result = client_objects[cid].run_local_eatac_tta(
                init_adapt_state=shared_adapter,
                test_loader=target_clients[[c["client_id"] for c in target_clients].index(cid)]["test_loader"],
                num_batches=configs.local_tta_batches,
                eval_after_adapt=configs.eval_after_adapt,
            )
            deltas.append(result["delta"])
            weights.append(result["num_samples"])
            per_result_accs.append(result["stats"]["accuracy"])

        # Single global average (all domains mixed — the problem)
        delta_avg = weighted_average_delta(deltas, weights)
        shared_adapter = apply_state_delta(shared_adapter, delta_avg, scale=1.0)

        # Round metrics from the SAME results (no repeated TTA)
        avg_acc_round = np.average(
            per_result_accs, weights=weights
        ) if weights else 0.0
        elapsed = time.time() - round_start

        round_metrics_list.append({
            "round": round_idx, "avg_acc": avg_acc_round,
            "num_clients": len(selected_cids),
        })
        logger.info(f"  Round {round_idx}: Acc={avg_acc_round:.2%} ({elapsed:.1f}s)")

    # Final evaluation on ALL clients (one more pass)
    final_eval = _final_eval_all_clients(
        target_clients, client_objects, shared_adapter, configs
    )

    return {"avg_acc": final_eval["avg_acc"], "avg_ece": final_eval["avg_ece"],
            "round_metrics": round_metrics_list}


def run_naive_fedacg_tta(target_clients, base_model, device, configs, fisher_values):
    """
    Baseline: Single shared adapter + momentum (all domains mixed).

    Used to prove that FedACG without domain isolation is still insufficient.
    """
    logger.info("=" * 60)
    logger.info("Baseline: Naive FedACG-TTA (no domain awareness)")
    logger.info("=" * 60)

    client_objects = {}
    for info in target_clients:
        client_objects[info["client_id"]] = FedTTAClient(
            client_id=info["client_id"], domain_id=info["domain_id"],
            corruption=info["corruption"], base_model=base_model,
            device=device, configs=configs, fisher_values=fisher_values,
        )

    source_state = extract_adapt_state(base_model)
    shared_adapter = copy.deepcopy(source_state)
    shared_momentum = zeros_like_state(source_state)
    lambda_tta = configs.lambda_tta
    tta_rounds = configs.tta_rounds
    join_ratio = getattr(configs, "target_participation_rate", 1.0)

    for round_idx in range(tta_rounds):
        # FedACG lookahead on shared adapter
        accelerated = add_state(shared_adapter, shared_momentum, scale=lambda_tta)

        num_to_sample = max(1, int(len(client_objects) * join_ratio))
        selected_cids = np.random.choice(
            list(client_objects.keys()), num_to_sample, replace=False
        ).tolist()

        deltas, weights, per_result_accs = [], [], []
        for cid in selected_cids:
            result = client_objects[cid].run_local_eatac_tta(
                init_adapt_state=accelerated,
                test_loader=target_clients[[c["client_id"] for c in target_clients].index(cid)]["test_loader"],
                num_batches=configs.local_tta_batches,
                eval_after_adapt=configs.eval_after_adapt,
            )
            deltas.append(result["delta"])
            weights.append(result["num_samples"])
            per_result_accs.append(result["stats"]["accuracy"])

        delta_avg = weighted_average_delta(deltas, weights)
        shared_momentum = add_state(scale_state(shared_momentum, lambda_tta), delta_avg, scale=1.0)
        shared_adapter = add_state(shared_adapter, shared_momentum, scale=1.0)

        avg_acc_round = np.average(per_result_accs, weights=weights) if weights else 0.0
        logger.info(f"  Round {round_idx}: Acc={avg_acc_round:.2%}")

    final_eval = _final_eval_all_clients(
        target_clients, client_objects, shared_adapter, configs
    )
    return final_eval


# ========== Domain-Isolated FedTTA (main methods) ==========


def run_domain_fedtta(
    target_clients, base_model, device, configs, fisher_values, use_momentum,
    save_path=None,
):
    """
    Domain-isolated FedTTA. Each corruption/domain has independent adapter.

    Args:
        use_momentum: True = FedACG, False = FedAvg
        save_path: directory for results. If None, creates one internally.
    """
    mode_name = "Domain FedACG-EATA-C" if use_momentum else "Domain FedAvg-EATA-C"
    logger.info("=" * 60)
    logger.info(f"Mode: {mode_name} (oracle domain grouping)")
    logger.info("=" * 60)

    # Create FedTTA server
    server = FedTTAServer(base_model, device, configs, use_momentum=use_momentum)

    # Create client objects
    client_objects = {}
    for info in target_clients:
        client_objects[info["client_id"]] = FedTTAClient(
            client_id=info["client_id"], domain_id=info["domain_id"],
            corruption=info["corruption"], base_model=base_model,
            device=device, configs=configs, fisher_values=fisher_values,
        )

    round_metrics_list = []
    domain_metrics_list = []
    per_client_metrics_list = []
    client_round_metrics_list = []

    # Resolve save path
    if save_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        task_name = getattr(configs, "task_name", "default")
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "fedtta", configs.dataset,
            f"{configs.tta_mode}_{configs.backbone}",
            f"{task_name}_{timestamp}",
        )
    os.makedirs(save_path, exist_ok=True)

    join_ratio = getattr(configs, "target_participation_rate", 1.0)

    # FedTTA rounds
    for round_idx in range(configs.tta_rounds):
        round_start = time.time()

        # Sample clients
        num_to_sample = max(1, int(len(client_objects) * join_ratio))
        selected_cids = np.random.choice(
            list(client_objects.keys()), num_to_sample, replace=False
        ).tolist()

        # Run round
        round_result = server.run_fedtta_round(
            target_clients=target_clients,
            clients_to_run=selected_cids,
            client_objects=client_objects,
        )

        elapsed = time.time() - round_start
        reliability_raw = round_result.get('avg_reliability_raw', 0)
        collapse = round_result.get('avg_collapse', 0)
        delta_n = round_result.get('avg_delta_norm', 0)
        comm_b = round_result.get('communication_bytes', 0)
        l_cons = round_result.get('avg_total_consistency_loss', 0)
        l_minmax = round_result.get('avg_total_minmax_loss', 0)
        l_fisher = round_result.get('avg_total_fisher_loss', 0)
        l_prox = round_result.get('avg_total_prox_loss', 0)
        l_total = round_result.get('avg_total_loss', 0)
        logger.info(f"  Round {round_idx}: "
                    f"Acc={round_result['avg_acc']:.2%} "
                    f"ECE={round_result['avg_ece']:.2%} "
                    f"Sel={round_result['avg_selected_ratio']:.1%} "
                    f"Rel={reliability_raw:.3f} "
                    f"Col={collapse:.3f} "
                    f"dN={delta_n:.4f} "
                    f"Lc={l_cons:.4f} Lm={l_minmax:.4f} "
                    f"Lf={l_fisher:.4f} Lp={l_prox:.4f} "
                    f"Lt={l_total:.4f} "
                    f"BW={round_result['num_backwards']} "
                    f"Comm={comm_b:,}B "
                    f"({elapsed:.1f}s)")

        rm = {"round": round_idx, **round_result}
        round_metrics_list.append(rm)

        # Accumulate per-client round stats
        for cr in round_result.get("client_round_stats", []):
            client_entry = {"round": round_idx, **cr}
            client_round_metrics_list.append(client_entry)

        for d_id, dm in round_result["domain_metrics"].items():
            # Log reliability weight distribution for this domain
            raw_scores = dm.get("raw_scores", [])
            agg_weights = dm.get("agg_weights", [])
            if raw_scores and agg_weights:
                n_clients = len(raw_scores)
                if n_clients > 1:
                    w_max = max(agg_weights)
                    w_min = min(agg_weights)
                    w_ratio = w_max / max(w_min, 1e-8)
                    w_entropy_val = -sum(w * np.log(max(w, 1e-15)) for w in agg_weights) / max(np.log(n_clients), 1e-8)
                    logger.info(f"    Domain {d_id} ({dm['corruption']}): "
                                f"weights range=[{w_min:.3f}, {w_max:.3f}] "
                                f"ratio={w_ratio:.1f}x "
                                f"norm_entropy={w_entropy_val:.3f}")
                else:
                    logger.info(f"    Domain {d_id} ({dm['corruption']}): single client, weight=1.0")

            domain_metrics_list.append({
                "round": round_idx, "domain_id": d_id,
                "corruption": dm["corruption"], "domain_acc": dm["domain_acc"],
                "domain_ece": dm["domain_ece"], "num_clients": dm["num_clients"],
                "num_samples": dm["num_samples"],
                "avg_selected_ratio": dm.get("avg_selected_ratio", 0),
                "avg_confidence": dm.get("avg_confidence", 0),
                "avg_entropy_norm": dm.get("avg_entropy_norm", 0),
                "avg_collapse": dm.get("avg_collapse", 0),
                "avg_reliability_raw": dm.get("avg_reliability_raw", 0),
                "avg_delta_norm": dm.get("avg_delta_norm", 0),
                "raw_scores": raw_scores,
                "agg_weights": agg_weights,
            })

    # Final evaluation on ALL clients (using evaluate_with_adapt_state, NO TTA)
    logger.info("=" * 60)
    logger.info("Final Evaluation — adapter_eval only (no TTA)")
    logger.info("=" * 60)

    # Ensure ALL domains are initialized (even those never selected in rounds)
    all_domain_ids = set()
    for ci in target_clients:
        if ci["domain_id"] not in all_domain_ids:
            all_domain_ids.add(ci["domain_id"])
            server._get_or_init_domain(ci["domain_id"], ci["corruption"])

    # Load clean test data for clean accuracy evaluation
    clean_dataset, _ = load_clean_test(configs)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=configs.test_batch_size,
        shuffle=False, num_workers=0,
    )

    adapter_eval_results = []
    for ci in target_clients:
        cid, d_id = ci["client_id"], ci["domain_id"]
        # Use server.domain_adapters[d_id] directly — NOT accelerated state
        adapter = server.domain_adapters[d_id]

        result = client_objects[cid].evaluate_with_adapt_state(
            adapt_state=adapter,
            test_loader=ci["test_loader"],
        )

        adapter_eval_results.append(result)
        per_client_metrics_list.append({
            "round": configs.tta_rounds - 1,
            "client_id": cid, "domain_id": d_id,
            "corruption": ci["corruption"],
            "adapter_eval_acc": result["accuracy"],
            "adapter_eval_ece": result["ece"],
            "num_samples": result["num_samples"],
        })

    # ---- Richer final metrics ----
    all_accs = [r["accuracy"] for r in adapter_eval_results]
    all_eces = [r["ece"] for r in adapter_eval_results]
    final_avg_acc = float(np.mean(all_accs))
    final_worst_acc = float(np.min(all_accs))
    final_std_acc = float(np.std(all_accs))
    final_avg_ece = float(np.mean(all_eces))

    # Communication bytes (sum over all rounds)
    total_comm_bytes = sum(
        rm.get("communication_bytes", 0) for rm in round_metrics_list
    )

    # Total forwards/backwards over all rounds
    total_fw = sum(rm.get("num_forwards", 0) for rm in round_metrics_list)
    total_bw = sum(rm.get("num_backwards", 0) for rm in round_metrics_list)

    # Delta norms (final adapter vs source adapter)
    delta_norms = []
    for d_id in sorted(server.domain_adapters.keys()):
        adapter = server.domain_adapters[d_id]
        src = server.source_adapt_state
        rms_list = []
        for k in adapter:
            if k in src:
                diff = adapter[k].cpu() - src[k].cpu()
                rms_list.append((diff ** 2).mean().item() ** 0.5)
        delta_norms.append(float(np.mean(rms_list)) if rms_list else 0.0)
    delta_norm_mean = float(np.mean(delta_norms)) if delta_norms else 0.0
    delta_norm_std = float(np.std(delta_norms)) if delta_norms else 0.0

    # ---- Clean accuracy evaluation over all domains ----
    clean_accs_per_domain = {}
    for d_id in sorted(server.domain_adapters.keys()):
        adapter = server.domain_adapters[d_id]
        # Take first client of this domain as representative
        for ci in target_clients:
            if ci["domain_id"] == d_id:
                result = client_objects[ci["client_id"]].evaluate_with_adapt_state(
                    adapt_state=adapter,
                    test_loader=clean_loader,
                )
                clean_accs_per_domain[d_id] = result["accuracy"]
                break

    # Source-only clean accuracy (baseline)
    source_model = copy.deepcopy(base_model).to(device)
    source_model.eval()
    src_logits_list, src_labels_list = [], []
    with torch.no_grad():
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            out = source_model(x)
            logits = out["logit"] if isinstance(out, dict) else out
            src_logits_list.append(logits.cpu())
            src_labels_list.append(y.cpu())
    src_logits = torch.cat(src_logits_list)
    src_labels = torch.cat(src_labels_list)
    source_clean_acc = compute_accuracy(src_logits, src_labels)

    clean_acc_drops = {
        d_id: source_clean_acc - clean_accs_per_domain[d_id]
        for d_id in clean_accs_per_domain
    }

    # ---- Log rich summary ----
    logger.info(f"  Adapter-Eval Avg Acc = {final_avg_acc:.2%}")
    logger.info(f"  Adapter-Eval Worst-Client Acc = {final_worst_acc:.2%}")
    logger.info(f"  Adapter-Eval Std Acc = {final_std_acc:.2%}")
    logger.info(f"  Adapter-Eval Avg ECE = {final_avg_ece:.2%}")
    logger.info(f"  Source Clean Acc = {source_clean_acc:.2%}")
    for d_id, drop in clean_acc_drops.items():
        logger.info(f"  Domain {d_id} Clean Acc Drop = {drop:.2%}")
    logger.info(f"  Delta Norm Mean = {delta_norm_mean:.6f}, Std = {delta_norm_std:.6f}")
    logger.info(f"  Total Communication = {total_comm_bytes:,} bytes")
    logger.info(f"  Total Forwards = {total_fw}, Backwards = {total_bw}")

    # Save results
    _save_results(save_path, round_metrics_list,
                  domain_metrics_list, per_client_metrics_list,
                  client_round_metrics_list, server, configs)

    return {
        "avg_acc": final_avg_acc,
        "worst_acc": final_worst_acc,
        "std_acc": final_std_acc,
        "avg_ece": final_avg_ece,
        "delta_norm_mean": delta_norm_mean,
        "delta_norm_std": delta_norm_std,
        "source_clean_acc": source_clean_acc,
        "clean_acc_drops": clean_acc_drops,
        "communication_bytes": total_comm_bytes,
        "num_forwards": total_fw,
        "num_backwards": total_bw,
        "round_metrics": round_metrics_list,
    }


def _final_eval_all_clients(target_clients, client_objects, adapter_state, configs):
    """Evaluate all target clients with given adapter state (no TTA)."""
    per_client = []
    domain_data = defaultdict(list)

    for ci in target_clients:
        result = client_objects[ci["client_id"]].evaluate_with_adapt_state(
            adapt_state=adapter_state,
            test_loader=ci["test_loader"],
        )
        per_client.append(result)
        domain_data[ci["domain_id"]].append(result)

    avg_acc = np.mean([r["accuracy"] for r in per_client])
    avg_ece = np.mean([r["ece"] for r in per_client])

    for d_id, results in domain_data.items():
        d_acc = np.mean([r["accuracy"] for r in results])
        corr = target_clients[[c["domain_id"] for c in target_clients].index(d_id)]["corruption"]
        logger.info(f"  Domain {d_id} ({corr}): Acc={d_acc:.2%}")

    return {"avg_acc": avg_acc, "avg_ece": avg_ece}


def _save_results(save_path, round_metrics, domain_metrics,
                  per_client_metrics, client_round_metrics, server, configs):
    """Save all experiment results to CSV and config to JSON."""
    os.makedirs(save_path, exist_ok=True)

    # Config
    config_dict = {k: str(v) for k, v in sorted(vars(configs).items())}
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Round metrics
    if round_metrics:
        with open(os.path.join(save_path, "round_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=round_metrics[0].keys())
            writer.writeheader()
            writer.writerows(round_metrics)

    # Domain metrics
    if domain_metrics:
        with open(os.path.join(save_path, "per_domain_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=domain_metrics[0].keys())
            writer.writeheader()
            writer.writerows(domain_metrics)

    # Per-client metrics
    if per_client_metrics:
        with open(os.path.join(save_path, "per_client_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_client_metrics[0].keys())
            writer.writeheader()
            writer.writerows(per_client_metrics)

    # Client-round metrics (per-client per-round stats including reliability)
    if client_round_metrics:
        with open(os.path.join(save_path, "client_round_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=client_round_metrics[0].keys())
            writer.writeheader()
            writer.writerows(client_round_metrics)

    # Domain adapters
    server.save_adapters(os.path.join(save_path, "final_domain_adapters.pt"))

    logger.info(f"  Results saved to: {save_path}")


# ========== Main ==========


def run():
    configs = parser.parse_args()
    configs.step = "fedtta"
    set_seed(configs.seed)

    # Handle alpha_calib / beta_fisher aliases
    if configs.alpha_calib != 0.1:
        configs.entropy_reg_weight = configs.alpha_calib
    if configs.beta_fisher != 50.0:
        configs.fisher_alpha = configs.beta_fisher
    # update_bn_stats overrides freeze_bn
    if getattr(configs, 'update_bn_stats', False):
        configs.freeze_bn = False

    device = torch.device(configs.device if torch.cuda.is_available() else "cpu")

    # Auto-detect num_class from checkpoint
    if configs.model_path and os.path.exists(configs.model_path):
        ckpt = torch.load(configs.model_path, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt:
                    sd = ckpt[key]
                    break
        if isinstance(sd, dict):
            for fc_key in ("fc.weight", "module.fc.weight"):
                if fc_key in sd:
                    inferred = sd[fc_key].shape[0]
                    if inferred != configs.num_class:
                        logger.info(f"  Auto-detected num_class={inferred} from checkpoint")
                        configs.num_class = inferred
                    break

    # ---- Setup save path ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_name = getattr(configs, "task_name", "default")
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "fedtta", configs.dataset,
        f"{configs.tta_mode}_{configs.backbone}",
        f"{task_name}_{timestamp}",
    )
    os.makedirs(save_path, exist_ok=True)
    add_file_handler(logger, save_path)

    logger.info("=" * 50)
    logger.info("FedTTA Configuration")
    logger.info("=" * 50)
    for key, value in sorted(vars(configs).items()):
        logger.info(f"  {key}: {value}")

    # ---- Load model ----
    logger.info(f"Loading model: {configs.backbone}")
    if configs.backbone in ("eatac_resnet18", "resnet18"):
        model = eatac_resnet18(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet34":
        model = eatac_resnet34(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet50":
        model = eatac_resnet50(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    elif configs.backbone == "eatac_resnet101":
        model = eatac_resnet101(
            num_classes=configs.num_class,
            subnet_drop_rate=configs.subnet_drop_rate,
        )
    else:
        raise ValueError(f"Unknown backbone: {configs.backbone}")

    if configs.model_path and os.path.exists(configs.model_path):
        load_checkpoint_weights(model, configs.model_path, device="cpu")
    model.to(device)

    # ---- Load Fisher ----
    fisher_values = None
    if configs.fisher_path and os.path.exists(configs.fisher_path):
        fisher_values = torch.load(configs.fisher_path, map_location="cpu")
        # Diagnostic: check Fisher keys vs model param names
        if isinstance(fisher_values, dict):
            model_param_names = {n for n, _ in model.named_parameters()}
            fisher_keys = set(fisher_values.keys())
            matched = fisher_keys & model_param_names
            if len(matched) < len(fisher_keys):
                missing = fisher_keys - model_param_names
                logger.warning(f"  Fisher key mismatch: {len(missing)} keys not in model")
                logger.warning(f"  Missing keys (first 10): {sorted(missing)[:10]}")
            nonzero = sum(v.count_nonzero().item() for v in fisher_values.values())
            logger.info(f"  Loaded Fisher: {len(fisher_values)} params, {nonzero} nonzero entries")

    # ---- Build target clients ----
    if configs.corruption is not None:
        corruptions = [configs.corruption]
    elif configs.corruption_list is not None:
        corruptions = configs.corruption_list
    else:
        corruptions = STANDARD_CORRUPTIONS
    logger.info(f"Building target clients for {len(corruptions)} corruptions...")
    logger.info(f"  Corruptions: {corruptions}")

    target_clients = build_single_corruption_target_clients(
        dataset_name=configs.dataset,
        corruptions=corruptions,
        severity=configs.severity,
        clients_per_corruption=configs.clients_per_corruption,
        samples_per_client=configs.samples_per_client,
        batch_size=configs.test_batch_size,
        label_skew=getattr(configs, "target_label_skew", False),
        alpha=getattr(configs, "target_alpha", 1.0),
        precomputed_c_path=getattr(configs, "precomputed_c_path", None),
        seed=configs.seed,
    )
    logger.info(f"  Total target clients: {len(target_clients)}")
    total_samples = sum(c["num_samples"] for c in target_clients)
    logger.info(f"  Total samples: {total_samples}")

    # ---- Dispatch to selected TTA mode ----
    if configs.tta_mode == "source_only":
        result = run_source_only(target_clients, model, device)  # kept for backward compat
        if result and "batch_stats_acc" in result:
            logger.info(f"  batch-stats Acc/ECE: {result['batch_stats_acc']:.2%} / {result['batch_stats_ece']:.2%}")

    elif configs.tta_mode == "local_tent":
        result = run_local_tent(target_clients, model, device, configs)

    elif configs.tta_mode == "local_eata":
        result = run_local_eata(target_clients, model, device, configs)

    elif configs.tta_mode == "local_eata_c":
        result = run_local_eata_c(target_clients, model, device, configs, fisher_values)

    elif configs.tta_mode == "naive_fedavg_tta":
        result = run_naive_fedavg_tta(
            target_clients, model, device, configs, fisher_values
        )

    elif configs.tta_mode == "naive_fedacg_tta":
        result = run_naive_fedacg_tta(
            target_clients, model, device, configs, fisher_values
        )

    elif configs.tta_mode == "domain_fedavg_eatac":
        result = run_domain_fedtta(
            target_clients, model, device, configs, fisher_values,
            use_momentum=False, save_path=save_path,
        )

    elif configs.tta_mode == "domain_fedacg_eatac":
        result = run_domain_fedtta(
            target_clients, model, device, configs, fisher_values,
            use_momentum=True, save_path=save_path,
        )

    else:
        raise ValueError(f"Unknown TTA mode: {configs.tta_mode}")

    logger.info("=" * 50)
    logger.info("FedTTA Complete!")
    logger.info("=" * 50)
    if result:
        logger.info("Final results summary:")
        for k, v in result.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: (dict)")
            elif isinstance(v, list):
                logger.info(f"  {k}: [{len(v)} entries]")
            elif isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    run()
