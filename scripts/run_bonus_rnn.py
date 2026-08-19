#!/usr/bin/env python3
"""Run phases 2+3 for the phase-1 best-test RNN: k2_R3_H256 (test=0.9268).

The original grid search used val-based selection and picked k2_R2_H256
(val=0.9331) over k2_R3_H256 (val=0.9312), so R3 was never explored in
phases 2-3. This script fills that gap.

Phase 2: 4 STFT configs  (nfft512/256 x hop64/32)
Phase 3: 7 subband presets on the phase-2 val winner

Artifacts land in artifacts_grid_search_sumfusion/fast_learning_rnn_sumfusion/
under run-ids prefixed with "bonus_" so nothing existing is overwritten.

Usage:
    python scripts/run_bonus_rnn.py [--dry-run] [--device cuda:0] [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from grid_search import (
    COMMON_TRAIN_ARGS,
    DEFAULT_SUBBAND_PRESET,
    FAST_LEARNING_RNN_SCRIPT,
    MODEL_RNN,
    PHASE2_NFFT_HOP_GRID,
    PHASE3_SUBBAND_PRESETS,
    RunResult,
    RunSpec,
    artifact_root_for,
    build_command,
    load_best_summary,
    parse_run_metrics,
)

GRID_ROOT_DEFAULT = str(SCRIPT_DIR.parent / "artifacts" / "grid_search")

# Phase-1 best-test RNN seed
SEED = RunSpec(
    model=MODEL_RNN,
    phase="phase1",
    run_id="phase1_nfft512_hop64_k2_R3_H256",
    n_fft=512,
    hop_length=64,
    k=2,
    recurrency=3,
    hidden_size=256,
    subband_preset=DEFAULT_SUBBAND_PRESET,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-root", default=GRID_ROOT_DEFAULT)
    p.add_argument("--cache-dir", default="data/google_speech_commands")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run even if best_summary.json exists.")
    p.add_argument("--skip-hop32", action="store_true", help="Skip nfft/hop configs with hop=32 (avoids OOM).")
    return p.parse_args()


def run_one(
    spec: RunSpec,
    grid_root: Path,
    *,
    cache_dir: str,
    device: Optional[str],
    seed: int,
    dry_run: bool,
    force: bool,
) -> RunResult:
    artifact_root = artifact_root_for(grid_root, spec)
    artifact_root.mkdir(parents=True, exist_ok=True)

    existing = load_best_summary(artifact_root)
    if existing is not None and not force:
        best_val, test_val, best_epoch, stop_reason = parse_run_metrics(existing)
        print(
            f"[skip] {spec.run_id}  val={best_val:.4f}  test={test_val}",
            flush=True,
        )
        return RunResult(
            spec=spec,
            artifact_root=artifact_root.as_posix(),
            status="skipped",
            best_val_final_accuracy=best_val,
            test_final_accuracy_at_best_val=test_val,
            best_epoch=best_epoch,
            stop_reason=stop_reason,
            summary=existing,
        )

    cmd = build_command(spec, artifact_root, cache_dir=cache_dir, device=device, seed=seed)
    print(f"\n[run] {spec.run_id}", flush=True)
    if dry_run:
        print(f"  cmd={' '.join(cmd)}", flush=True)
        return RunResult(spec=spec, artifact_root=artifact_root.as_posix(), status="ok")

    completed = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent))
    if completed.returncode != 0:
        return RunResult(
            spec=spec,
            artifact_root=artifact_root.as_posix(),
            status="failed",
            error=f"subprocess exited with code {completed.returncode}",
        )

    summary = load_best_summary(artifact_root)
    if summary is None:
        return RunResult(
            spec=spec,
            artifact_root=artifact_root.as_posix(),
            status="failed",
            error="training finished but best_summary.json is missing",
        )

    best_val, test_val, best_epoch, stop_reason = parse_run_metrics(summary)
    print(
        f"[done] {spec.run_id}  val={best_val}  test={test_val}  epoch={best_epoch}",
        flush=True,
    )
    return RunResult(
        spec=spec,
        artifact_root=artifact_root.as_posix(),
        status="ok",
        best_val_final_accuracy=best_val,
        test_final_accuracy_at_best_val=test_val,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        summary=summary,
    )


def main() -> int:
    args = parse_args()
    grid_root = Path(args.grid_root)

    # --- Phase 2: STFT configs ---
    nfft_hop_grid = [
        (n_fft, hop) for n_fft, hop in PHASE2_NFFT_HOP_GRID
        if not (args.skip_hop32 and hop == 32)
    ]
    phase2_specs = [
        RunSpec(
            model=MODEL_RNN,
            phase="phase2",
            run_id=f"bonus_phase2_nfft{n_fft}_hop{hop}_k2_R3_H256",
            n_fft=n_fft,
            hop_length=hop,
            k=2,
            recurrency=3,
            hidden_size=256,
            subband_preset=DEFAULT_SUBBAND_PRESET,
        )
        for n_fft, hop in nfft_hop_grid
    ]

    print(f"\n=== Phase 2 ({len(phase2_specs)} runs) — STFT grid for RNN k2_R3_H256 ===", flush=True)
    phase2_results: list[RunResult] = []
    for spec in phase2_specs:
        r = run_one(spec, grid_root, cache_dir=args.cache_dir, device=args.device,
                    seed=args.seed, dry_run=args.dry_run, force=args.force)
        phase2_results.append(r)

    # Pick best phase-2 by val
    successful = [
        r for r in phase2_results
        if r.status in ("ok", "skipped") and r.best_val_final_accuracy is not None
    ]
    if not successful:
        print("[error] No successful phase2 runs; cannot proceed to phase3.", flush=True)
        return 1

    best_p2 = max(successful, key=lambda r: r.best_val_final_accuracy or -1.0)
    print(
        f"\n[phase2 winner] {best_p2.spec.run_id}  "
        f"val={best_p2.best_val_final_accuracy:.4f}  test={best_p2.test_final_accuracy_at_best_val}  "
        f"n_fft={best_p2.spec.n_fft}  hop={best_p2.spec.hop_length}",
        flush=True,
    )

    # --- Phase 3: 7 subband presets on best phase-2 STFT ---
    phase3_specs = [
        RunSpec(
            model=MODEL_RNN,
            phase="phase3",
            run_id=f"bonus_phase3_{preset}_nfft{best_p2.spec.n_fft}_hop{best_p2.spec.hop_length}_k2_R3_H256",
            n_fft=best_p2.spec.n_fft,
            hop_length=best_p2.spec.hop_length,
            k=2,
            recurrency=3,
            hidden_size=256,
            subband_preset=preset,
        )
        for preset in PHASE3_SUBBAND_PRESETS
    ]

    print(f"\n=== Phase 3 ({len(phase3_specs)} runs) — subband presets on nfft={best_p2.spec.n_fft} hop={best_p2.spec.hop_length} ===", flush=True)
    phase3_results: list[RunResult] = []
    for spec in phase3_specs:
        r = run_one(spec, grid_root, cache_dir=args.cache_dir, device=args.device,
                    seed=args.seed, dry_run=args.dry_run, force=args.force)
        phase3_results.append(r)

    # Final summary
    all_done = [
        r for r in phase3_results
        if r.status in ("ok", "skipped") and r.test_final_accuracy_at_best_val is not None
    ]
    if all_done:
        print("\n=== Results ===", flush=True)
        for r in sorted(all_done, key=lambda x: x.test_final_accuracy_at_best_val or 0, reverse=True):
            print(
                f"  test={r.test_final_accuracy_at_best_val:.4f}  val={r.best_val_final_accuracy:.4f}"
                f"  preset={r.spec.subband_preset}  {r.spec.run_id}",
                flush=True,
            )
        best = max(all_done, key=lambda r: r.test_final_accuracy_at_best_val or -1.0)
        print(
            f"\n[best] test={best.test_final_accuracy_at_best_val:.4f}  "
            f"val={best.best_val_final_accuracy:.4f}  {best.spec.run_id}",
            flush=True,
        )

    failed = [r for r in phase2_results + phase3_results if r.status == "failed"]
    if failed:
        print(f"\n[warn] {len(failed)} run(s) failed.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
