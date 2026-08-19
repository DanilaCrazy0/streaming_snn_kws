#!/usr/bin/env python3
"""Train the published RNN configuration (test accuracy 0.9268)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "snn_kws" / "train_rnn.py"

BEST_ARGS = [
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
    "--artifact-root", "artifacts/best_rnn",
]


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *BEST_ARGS, *sys.argv[1:]]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
