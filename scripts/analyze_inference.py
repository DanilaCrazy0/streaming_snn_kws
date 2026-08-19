#!/usr/bin/env python
"""CLI entry: run SumFusion inference-only analysis (journal memory/spike analyses)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_inference_utils import OUT_DIR, run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="SumFusion inference analysis (no retrain)")
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-layered-memory", action="store_true", help="Skip layered E3a reset pass")
    parser.add_argument(
        "--freeze-fracs",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="E3b freeze fractions of T (default: 0.25 0.5 0.75, as in the article)",
    )
    args = parser.parse_args()

    print(f"Repo: {REPO_ROOT}")
    print(f"Output: {OUT_DIR}")
    summary = run_all(
        val_limit=args.val_limit,
        batch_size=args.batch_size,
        device=args.device,
        run_layered_memory=not args.no_layered_memory,
        freeze_fracs=tuple(args.freeze_fracs),
    )
    report = OUT_DIR / "REPORT.md"
    js = OUT_DIR / "report_summary.json"
    assert report.is_file(), f"Missing {report}"
    assert js.is_file(), f"Missing {js}"
    e3a = summary.get("E3a_memory_ablation", {})
    print("E3a keys:", list(e3a.keys()))
    for arch, d in e3a.items():
        print(
            f"  {arch}: streaming={d.get('streaming', {}).get('final_accuracy')}, "
            f"reset={d.get('reset', {}).get('final_accuracy')}, "
            f"delta={d.get('delta_final_accuracy')}"
        )
    e3b = summary.get("E3b_frozen_input", {})
    print("E3b by_frac:", e3b.get("by_frac"))
    print("Figures:", summary.get("figures"))
    print("DONE")


if __name__ == "__main__":
    main()
