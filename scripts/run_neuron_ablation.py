#!/usr/bin/env python3
"""Run LIF then AdLIF on the published streaming KWS recipe (RTX 4090).

This is the neuron ablation for the GSN paper: same STFT, subbands, k, R, H,
fusion, losses, scheduler, seed, CUDA graphs. Only the membrane cell changes.

Default (RNN, published winner, ~2 x 30 min on 4090):

    python scripts/run_neuron_ablation.py --device cuda:0

That trains LIF, then AdLIF, then writes
``artifacts/ablation_rnn_neurons/comparison.json`` against the published GSU
numbers (val 0.9312 / test 0.9268). GSU is not retrained unless
``--include-gsu`` is set.

If a run already has ``best_summary.json``, it is skipped (resume-safe).
Pass ``--force`` to retrain. Extra args after ``--`` are forwarded, e.g.

    python scripts/run_neuron_ablation.py --device cuda:0 -- --batch-size 400
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_RNN = REPO_ROOT / "snn_kws" / "train_rnn.py"
TRAIN_LAYERED = REPO_ROOT / "snn_kws" / "train_layered.py"

PUBLISHED_GSU_RNN = {
    "neuron_type": "gsu",
    "source": "thesis_published",
    "val_final_accuracy": 0.9312,
    "test_final_accuracy": 0.9268,
    "num_params": 1718051,
    "epochs": 30,
    "batch_size": 500,
}

PUBLISHED_GSU_LAYERED = {
    "neuron_type": "gsu",
    "source": "thesis_published_extended",
    "val_final_accuracy": 0.9347,
    "test_final_accuracy": 0.9257,
    "epochs": 65,
    "batch_size": 256,
}

RNN_BEST = [
    "--k", "2",
    "--recurrency", "3",
    "--hidden-size", "256",
    "--n-fft", "512",
    "--hop-length", "64",
    "--subband-preset", "p3_default",
    "--batch-size", "500",
    "--epochs", "30",
    "--seed", "7",
    "--lr-scheduler", "gated_plateau",
    "--lr-gate-accuracy", "0.8",
    "--plateau-factor", "0.5",
    "--plateau-patience", "2",
    "--plateau-threshold", "1e-3",
    "--plateau-threshold-mode", "abs",
    "--min-learning-rate", "2e-5",
    "--final-weight", "1.0",
    "--prefix-weight", "0.75",
    "--consistency-weight", "0.15",
    "--cuda-graph",
]

LAYERED_BEST = [
    "--k", "2",
    "--p", "1",
    "--branch-layers", "2",
    "--hidden-size", "256",
    "--n-fft", "256",
    "--hop-length", "64",
    "--subband-preset", "p7_full_0_2_2_5_5_8",
    "--batch-size", "256",
    "--epochs", "65",
    "--seed", "7",
    "--lr-scheduler", "phased",
    "--lr-gate-accuracy", "0.89",
    "--lr-plateau-until-accuracy", "0.92",
    "--plateau-factor", "0.5",
    "--plateau-patience", "2",
    "--plateau-threshold", "1e-3",
    "--plateau-threshold-mode", "abs",
    "--min-learning-rate", "3e-5",
    "--final-weight", "1.0",
    "--prefix-weight", "0.75",
    "--consistency-weight", "0.15",
    "--cuda-graph",
]


def _artifact_root(architecture: str, neuron: str) -> Path:
    return REPO_ROOT / "artifacts" / f"ablation_{architecture}_{neuron}"


def _run_one(
    *,
    architecture: str,
    neuron: str,
    device: Optional[str],
    extra_args: list[str],
    force: bool,
    dry_run: bool,
) -> int:
    artifact_root = _artifact_root(architecture, neuron)
    summary_path = artifact_root / "best_summary.json"
    if summary_path.is_file() and not force:
        print(f"[skip] {neuron} {architecture}: {summary_path} already exists", flush=True)
        return 0

    script = TRAIN_RNN if architecture == "rnn" else TRAIN_LAYERED
    base = RNN_BEST if architecture == "rnn" else LAYERED_BEST
    cmd = [
        sys.executable,
        str(script),
        *base,
        "--neuron-type",
        neuron,
        "--artifact-root",
        str(artifact_root.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]
    if device:
        cmd.extend(["--device", device])
    cmd.extend(extra_args)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _read_run(architecture: str, neuron: str) -> Optional[dict[str, Any]]:
    path = _artifact_root(architecture, neuron) / "best_summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    return {
        "neuron_type": neuron,
        "source": "this_run",
        "artifact_root": str(_artifact_root(architecture, neuron)),
        "val_final_accuracy": summary.get("best_val_final_accuracy"),
        "test_final_accuracy": summary.get("test_final_accuracy_at_best_val"),
        "best_epoch": summary.get("best_epoch"),
        "num_params": summary.get("num_params"),
        "cuda_graph": summary.get("cuda_graph"),
        "summary": summary,
        "config": payload.get("config"),
    }


def _write_comparison(architecture: str) -> Path:
    published = PUBLISHED_GSU_RNN if architecture == "rnn" else PUBLISHED_GSU_LAYERED
    rows = [published]
    for neuron in ("gsu", "lif", "adlif"):
        row = _read_run(architecture, neuron)
        if row is not None:
            rows.append(row)

    gsu_test = float(published["test_final_accuracy"])
    table = []
    for row in rows:
        test = row.get("test_final_accuracy")
        delta = None
        if isinstance(test, (int, float)) and test == test:
            delta = float(test) - gsu_test
        table.append(
            {
                "neuron_type": row["neuron_type"],
                "source": row["source"],
                "val_final_accuracy": row.get("val_final_accuracy"),
                "test_final_accuracy": test,
                "delta_test_vs_published_gsu": delta,
                "num_params": row.get("num_params"),
            }
        )

    out_dir = REPO_ROOT / "artifacts" / f"ablation_{architecture}_neurons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.json"
    payload = {
        "architecture": architecture,
        "note": (
            "GSU is the published gated cell. LIF and AdLIF keep the same "
            "streaming fusion architecture; a drop in test accuracy is the "
            "GSN contribution under this recipe."
        ),
        "published_gsu": published,
        "runs": rows,
        "table": table,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[comparison] {out_path}", flush=True)
    print(
        f"{'neuron':<10} {'source':<24} {'val':>8} {'test':>8} {'d_test':>8} {'params':>10}",
        flush=True,
    )
    for item in table:
        val = item["val_final_accuracy"]
        test = item["test_final_accuracy"]
        delta = item["delta_test_vs_published_gsu"]
        params = item["num_params"]
        val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
        test_s = f"{test:.4f}" if isinstance(test, float) else str(test)
        delta_s = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
        params_s = str(params) if params is not None else "-"
        print(
            f"{item['neuron_type']:<10} {item['source']:<24} {val_s:>8} {test_s:>8} {delta_s:>8} {params_s:>10}",
            flush=True,
        )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LIF then AdLIF on the published streaming KWS recipe."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--architecture",
        default="rnn",
        choices=("rnn", "layered", "both"),
        help="RNN is the published best (test 0.9268). Layered is the 65-epoch GSN-branch recipe.",
    )
    parser.add_argument(
        "--neurons",
        nargs="+",
        default=["lif", "adlif"],
        choices=("gsu", "lif", "adlif"),
        help="Order of runs. Default: lif then adlif.",
    )
    parser.add_argument(
        "--include-gsu",
        action="store_true",
        help="Also retrain GSU on this GPU (fair hardware control). Default uses published numbers.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if best_summary.json exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Extra flags after -- are passed to the train script (e.g. --batch-size 400).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra = list(args.forwarded)
    if extra and extra[0] == "--":
        extra = extra[1:]

    architectures = ["rnn", "layered"] if args.architecture == "both" else [args.architecture]
    neurons = list(args.neurons)
    if args.include_gsu and "gsu" not in neurons:
        neurons = ["gsu", *neurons]

    print(
        {
            "stage": "ablation_plan",
            "architectures": architectures,
            "neurons": neurons,
            "device": args.device,
            "note": "Same recipe as the published GSN model; only --neuron-type changes.",
        },
        flush=True,
    )

    failures = []
    for architecture in architectures:
        for neuron in neurons:
            code = _run_one(
                architecture=architecture,
                neuron=neuron,
                device=args.device,
                extra_args=extra,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
            if code != 0:
                failures.append((architecture, neuron, code))
                print(
                    f"[error] {architecture} {neuron} exited with {code}; continuing.",
                    flush=True,
                )
        if not args.dry_run:
            _write_comparison(architecture)

    if failures:
        print(f"[done] failures={failures}", flush=True)
        raise SystemExit(1)
    print("[done] neuron ablation finished", flush=True)


if __name__ == "__main__":
    main()
