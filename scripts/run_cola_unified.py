"""Run ETA+CoLA with the FEDEATA unified Tiny-ImageNet-C protocol.

The wrapper keeps the method implementation in ``main_test.py`` while making
the cross-method protocol explicit: every corruption is given the same
RandomState(0) permutation, the resolved CoLA configuration is recorded, and
full runs are rejected unless all expected samples are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


EXPECTED_CORRUPTIONS = 15
EXPECTED_CONTINUAL_DOMAINS = 45
EXPECTED_SAMPLES_PER_DOMAIN = 10_000


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CoLA baseline under the FEDEATA unified protocol."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for order audits, resolved config, and validation files.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow --max_batches smoke runs without full-sample validation.",
    )
    parser.add_argument(
        "main_test_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to main_test.py after '--'.",
    )
    parsed = parser.parse_args()
    if parsed.main_test_args[:1] == ["--"]:
        parsed.main_test_args = parsed.main_test_args[1:]
    return parsed


def _argument_value(arguments, flag):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"main_test arguments must include {flag}") from exc


def _install_order_audit(output_dir: Path):
    from evaluator.dataset import TinyImageNetCDataset

    original_init = TinyImageNetCDataset.__init__

    def ordered_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if len(self.samples) != EXPECTED_SAMPLES_PER_DOMAIN:
            raise AssertionError(
                "Tiny-ImageNet-C domain has "
                f"{len(self.samples)} samples; expected {EXPECTED_SAMPLES_PER_DOMAIN}"
            )
        permutation = np.random.RandomState(0).permutation(len(self.samples))
        self.samples = [self.samples[int(index)] for index in permutation]

        records = [
            [f"{Path(path).parent.name}/{Path(path).name}", int(target)]
            for path, target in self.samples
        ]
        digest = hashlib.sha256(
            json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        corruption = kwargs.get("corruption")
        if corruption is None and len(args) >= 2:
            corruption = args[1]
        if corruption is None:
            raise AssertionError("could not identify Tiny-ImageNet-C corruption")

        audit = {
            "corruption": str(corruption),
            "seed": 0,
            "order": "RandomState(0) permutation of sorted class/file list",
            "count": len(records),
            "sha256": digest,
            "first_samples": records[:5],
            "last_samples": records[-5:],
        }
        audit_path = output_dir / "sample_orders" / f"{corruption}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            if existing != audit:
                raise AssertionError(f"sample order changed for {corruption}")
        else:
            audit_path.write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    TinyImageNetCDataset.__init__ = ordered_init


def _validate_result(result_path: Path, output_dir: Path, allow_partial: bool):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    domain = result["domain_wise"]["per_corruption"]
    continual = result["continual"]["per_corruption"]
    post_stream = result["continual"]["post_stream"]
    if len(domain) != EXPECTED_CORRUPTIONS:
        raise AssertionError(f"expected 15 domain-wise rows, got {len(domain)}")
    if len(continual) != EXPECTED_CONTINUAL_DOMAINS:
        raise AssertionError(f"expected 45 continual rows, got {len(continual)}")

    if not allow_partial:
        rows = domain + continual
        if any(row["total"] != EXPECTED_SAMPLES_PER_DOMAIN for row in rows):
            raise AssertionError("full protocol did not evaluate 10,000 samples/domain")
        if post_stream["clean"]["total"] != EXPECTED_SAMPLES_PER_DOMAIN:
            raise AssertionError("clean post-stream evaluation is incomplete")
        if len(post_stream["revisit"]) != 4:
            raise AssertionError("expected four post-stream revisit domains")
        if any(
            row["total"] != EXPECTED_SAMPLES_PER_DOMAIN
            for row in post_stream["revisit"]
        ):
            raise AssertionError("revisit evaluation is incomplete")

    protocol = result.setdefault("protocol", {})
    protocol["within_domain_order"] = (
        "RandomState(0) permutation of sorted class/file list, repeated identically"
    )
    protocol["order_audit_dir"] = str(output_dir / "sample_orders")
    protocol["batch_size"] = 128
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "cola_resolved_config.json").write_text(
        json.dumps(result["config"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    validation = {
        "complete": not allow_partial,
        "method": result.get("method"),
        "public_tta_entrypoint": result.get("public_tta_entrypoint"),
        "domains": len(domain),
        "continual_domains": len(continual),
        "samples_per_domain": EXPECTED_SAMPLES_PER_DOMAIN,
        "clean_samples": post_stream["clean"]["total"],
        "revisit_domains": len(post_stream["revisit"]),
        "config_hash": result.get("config_hash"),
    }
    (output_dir / "cola_validated.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parsed = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = parsed.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    arguments = list(parsed.main_test_args)
    if "--output_path" not in arguments:
        arguments.extend(["--output_path", str(output_dir / "cola.json")])
    result_path = Path(_argument_value(arguments, "--output_path"))
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    result_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    _install_order_audit(output_dir)
    module = importlib.import_module("main_test")

    original_cola_config = module.apply_cola_primary_config

    def recorded_cola_config(args):
        args = original_cola_config(args)
        (output_dir / "cola_resolved_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return args

    module.apply_cola_primary_config = recorded_cola_config
    sys.argv = ["main_test.py"] + arguments
    module.run()
    _validate_result(result_path, output_dir, parsed.allow_partial)


if __name__ == "__main__":
    main()
