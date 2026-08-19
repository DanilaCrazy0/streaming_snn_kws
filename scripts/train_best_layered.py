#!/usr/bin/env python3
"""Train the published layered GSU-branch configuration (test accuracy 0.9257)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "snn_kws" / "train_layered.py"

BEST_ARGS = [
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
    "--artifact-root", "artifacts/best_layered",
]


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *BEST_ARGS, *sys.argv[1:]]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
