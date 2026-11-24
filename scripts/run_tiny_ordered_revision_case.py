"""Run the revision entrypoint with the unified Tiny-ImageNet-C stream order."""

import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: run_tiny_ordered_revision_case.py RUN_CASE_PATH OUTPUT_DIR [args...]"
        )
    run_case = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    forwarded = sys.argv[3:]
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(run_case.parents[2]))

    from evaluator.dataset import TinyImageNetCDataset

    original_init = TinyImageNetCDataset.__init__

    def ordered_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if len(self.samples) != 10000:
            raise ValueError(f"expected 10,000 Tiny-C samples, got {len(self.samples)}")
        order = np.random.RandomState(0).permutation(len(self.samples))
        self.samples = [self.samples[int(i)] for i in order]
        records = [
            [f"{Path(path).parent.name}/{Path(path).name}", int(label)]
            for path, label in self.samples
        ]
        corruption = kwargs["corruption"] if "corruption" in kwargs else args[1]
        payload = {
            "corruption": corruption,
            "seed": 0,
            "count": len(records),
            "sha256": hashlib.sha256(json.dumps(records).encode("utf-8")).hexdigest(),
            "ordered_samples": records,
        }
        audit_dir = output_dir / "sample_orders"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{payload['corruption']}.json"
        if audit_path.exists():
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
            if previous != payload:
                raise ValueError(f"Tiny-C sample order changed for {payload['corruption']}")
        else:
            audit_path.write_text(json.dumps(payload), encoding="utf-8")

    TinyImageNetCDataset.__init__ = ordered_init
    sys.argv = [str(run_case), *forwarded]
    try:
        runpy.run_path(str(run_case), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    output_arg = forwarded[forwarded.index("--output_path") + 1]
    result_path = Path(output_arg)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = result.setdefault("protocol", {})
    protocol["within_domain_order"] = (
        "RandomState(0) permutation of sorted class/file list, repeated identically"
    )
    protocol["sample_order_sha256"] = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["sha256"]
        for path in sorted((output_dir / "sample_orders").glob("*.json"))
    }
    domain = result["domain_wise"]["per_corruption"]
    continual = result["continual"]["per_corruption"]
    revisit = result["continual"]["post_stream"]["revisit"]
    clean = result["continual"]["post_stream"]["clean"]
    if (
        len(domain) != 15
        or len(continual) != 45
        or any(row["total"] != 10000 for row in domain + continual)
        or clean["total"] != 10000
        or len(revisit) != 4
        or any(row["total"] != 10000 for row in revisit)
        or len(protocol["sample_order_sha256"]) != 15
    ):
        raise ValueError("Tiny-C result failed full-protocol coverage validation")
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
