#!/usr/bin/env python3
"""Three-phase grid search for sum-threshold pre-fusion + GSN fusion_head.

Phase 1: architecture hyperparams at n_fft=512, hop=64.
Phase 2: STFT grid from phase-1 winners.
Phase 3: subband presets from phase-2 winners.

Runs training subprocesses one at a time, alternating architectures within each
phase. Writes incremental grid_summary.json after every run for crash resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAST_LEARNING_SCRIPT = REPO_ROOT / "snn_kws" / "train_layered.py"
FAST_LEARNING_RNN_SCRIPT = REPO_ROOT / "snn_kws" / "train_rnn.py"

MODEL_BRANCHES = "fast_learning_sumfusion"
MODEL_RNN = "fast_learning_rnn_sumfusion"

PHASE1_NFFT = 512
PHASE1_HOP = 64
DEFAULT_SUBBAND_PRESET = "p3_default"

PHASE2_NFFT_HOP_GRID: tuple[tuple[int, int], ...] = (
    (512, 64),
    (512, 32),
    (256, 64),
    (256, 32),
)

PHASE3_SUBBAND_PRESETS: tuple[str, ...] = (
    "p1_full_0_4_4_8",
    "p2_full_0_2_2_4_4_6_6_8",
    "p3_default",
    "p4_full_0_1_1_8",
    "p5_full_0_4_4_8",
    "p6_full_0_3_3_6_6_8",
    "p7_full_0_2_2_5_5_8",
)

COMMON_TRAIN_ARGS: tuple[str, ...] = (
    "--batch-size",
    "500",
    "--epochs",
    "30",
    "--target-accuracy",
    "0.95",
    "--num-workers",
    "8",
    "--cuda-graph",
    "--final-weight",
    "1.0",
    "--prefix-weight",
    "0.75",
    "--consistency-weight",
    "0.15",
    "--plateau-patience",
    "2",
    "--plateau-factor",
    "0.5",
    "--plateau-threshold",
    "1e-3",
    "--plateau-threshold-mode",
    "abs",
)


@dataclass
class RunSpec:
    model: str
    phase: str
    run_id: str
    n_fft: int
    hop_length: int
    k: int = 0
    p: int = 1
    recurrency: int = 2
    branch_layers: int = 2
    hidden_size: int = 128
    subband_preset: str = DEFAULT_SUBBAND_PRESET
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RunResult:
    spec: RunSpec
    artifact_root: str
    status: str
    best_val_final_accuracy: Optional[float] = None
    test_final_accuracy_at_best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "model": self.spec.model,
            "phase": self.spec.phase,
            "run_id": self.spec.run_id,
            "n_fft": self.spec.n_fft,
            "hop_length": self.spec.hop_length,
            "k": self.spec.k,
            "p": self.spec.p,
            "recurrency": self.spec.recurrency,
            "branch_layers": self.spec.branch_layers,
            "hidden_size": self.spec.hidden_size,
            "subband_preset": self.spec.subband_preset,
            "artifact_root": self.artifact_root,
            "status": self.status,
            "best_val_final_accuracy": self.best_val_final_accuracy,
            "test_final_accuracy_at_best_val": self.test_final_accuracy_at_best_val,
            "best_epoch": self.best_epoch,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }
        if self.summary is not None:
            out["summary"] = self.summary
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-phase grid search for sumfusion branch/RNN architectures."
    )
    parser.add_argument(
        "--grid-root",
        default=str(REPO_ROOT / "artifacts" / "grid_search"),
        help="Root directory for per-run artifacts and grid_summary.json.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/google_speech_commands",
        help="Dataset cache passed through to training scripts.",
    )
    parser.add_argument("--device", default=None, help="Torch device for training (e.g. cuda:0).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without executing training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when best_summary.json already exists.",
    )
    return parser.parse_args()


def build_branches_phase1_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for k in (0, 1, 2):
        for branch_layers in (2, 3, 4):
            for hidden_size in (128, 256):
                run_id = f"phase1_nfft{PHASE1_NFFT}_hop{PHASE1_HOP}_k{k}_L{branch_layers}_H{hidden_size}"
                specs.append(
                    RunSpec(
                        model=MODEL_BRANCHES,
                        phase="phase1",
                        run_id=run_id,
                        n_fft=PHASE1_NFFT,
                        hop_length=PHASE1_HOP,
                        k=k,
                        p=1,
                        branch_layers=branch_layers,
                        hidden_size=hidden_size,
                        subband_preset=DEFAULT_SUBBAND_PRESET,
                    )
                )
    return specs


def build_rnn_phase1_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for k in (0, 1, 2):
        for recurrency in (2, 3, 4):
            for hidden_size in (128, 256):
                run_id = f"phase1_nfft{PHASE1_NFFT}_hop{PHASE1_HOP}_k{k}_R{recurrency}_H{hidden_size}"
                specs.append(
                    RunSpec(
                        model=MODEL_RNN,
                        phase="phase1",
                        run_id=run_id,
                        n_fft=PHASE1_NFFT,
                        hop_length=PHASE1_HOP,
                        k=k,
                        recurrency=recurrency,
                        hidden_size=hidden_size,
                        subband_preset=DEFAULT_SUBBAND_PRESET,
                    )
                )
    return specs


def build_phase2_specs(best: RunSpec) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for n_fft, hop in PHASE2_NFFT_HOP_GRID:
        if best.model == MODEL_BRANCHES:
            run_id = (
                f"phase2_nfft{n_fft}_hop{hop}_k{best.k}_L{best.branch_layers}_H{best.hidden_size}"
            )
            specs.append(
                RunSpec(
                    model=MODEL_BRANCHES,
                    phase="phase2",
                    run_id=run_id,
                    n_fft=n_fft,
                    hop_length=hop,
                    k=best.k,
                    p=best.p,
                    branch_layers=best.branch_layers,
                    hidden_size=best.hidden_size,
                    subband_preset=DEFAULT_SUBBAND_PRESET,
                )
            )
        else:
            run_id = (
                f"phase2_nfft{n_fft}_hop{hop}_k{best.k}_R{best.recurrency}_H{best.hidden_size}"
            )
            specs.append(
                RunSpec(
                    model=MODEL_RNN,
                    phase="phase2",
                    run_id=run_id,
                    n_fft=n_fft,
                    hop_length=hop,
                    k=best.k,
                    recurrency=best.recurrency,
                    hidden_size=best.hidden_size,
                    subband_preset=DEFAULT_SUBBAND_PRESET,
                )
            )
    return specs


def build_phase3_specs(best: RunSpec) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for preset in PHASE3_SUBBAND_PRESETS:
        if best.model == MODEL_BRANCHES:
            run_id = (
                f"phase3_{preset}_nfft{best.n_fft}_hop{best.hop_length}_"
                f"k{best.k}_L{best.branch_layers}_H{best.hidden_size}"
            )
            specs.append(
                RunSpec(
                    model=MODEL_BRANCHES,
                    phase="phase3",
                    run_id=run_id,
                    n_fft=best.n_fft,
                    hop_length=best.hop_length,
                    k=best.k,
                    p=best.p,
                    branch_layers=best.branch_layers,
                    hidden_size=best.hidden_size,
                    subband_preset=preset,
                )
            )
        else:
            run_id = (
                f"phase3_{preset}_nfft{best.n_fft}_hop{best.hop_length}_"
                f"k{best.k}_R{best.recurrency}_H{best.hidden_size}"
            )
            specs.append(
                RunSpec(
                    model=MODEL_RNN,
                    phase="phase3",
                    run_id=run_id,
                    n_fft=best.n_fft,
                    hop_length=best.hop_length,
                    k=best.k,
                    recurrency=best.recurrency,
                    hidden_size=best.hidden_size,
                    subband_preset=preset,
                )
            )
    return specs


def interleave_specs(left: list[RunSpec], right: list[RunSpec]) -> list[RunSpec]:
    out: list[RunSpec] = []
    i = j = 0
    turn_left = True
    while i < len(left) and j < len(right):
        if turn_left:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
        turn_left = not turn_left
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def artifact_root_for(grid_root: Path, spec: RunSpec) -> Path:
    return grid_root / spec.model / spec.run_id


def build_command(
    spec: RunSpec,
    artifact_root: Path,
    *,
    cache_dir: str,
    device: Optional[str],
    seed: int,
) -> list[str]:
    script = FAST_LEARNING_SCRIPT if spec.model == MODEL_BRANCHES else FAST_LEARNING_RNN_SCRIPT
    cmd: list[str] = [
        sys.executable,
        str(script),
        "--artifact-root",
        str(artifact_root),
        "--log-file",
        str(artifact_root / "train.log"),
        "--cache-dir",
        cache_dir,
        "--seed",
        str(seed),
        "--n-fft",
        str(spec.n_fft),
        "--hop-length",
        str(spec.hop_length),
        "--k",
        str(spec.k),
        "--hidden-size",
        str(spec.hidden_size),
        "--fusion-hidden-size",
        str(spec.hidden_size),
        "--subband-preset",
        spec.subband_preset,
        *COMMON_TRAIN_ARGS,
    ]
    if spec.model == MODEL_BRANCHES:
        cmd.extend(["--p", str(spec.p), "--branch-layers", str(spec.branch_layers)])
    else:
        cmd.extend(["--recurrency", str(spec.recurrency)])
    if device is not None:
        cmd.extend(["--device", device])
    cmd.extend(spec.extra_args)
    return cmd


def load_best_summary(artifact_root: Path) -> Optional[dict[str, Any]]:
    path = artifact_root / "best_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_run_metrics(
    summary: dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[str]]:
    inner = summary.get("summary") or {}
    best_val = inner.get("best_val_final_accuracy")
    test_val = inner.get("test_final_accuracy_at_best_val")
    best_epoch = inner.get("best_epoch")
    stop_reason = inner.get("stop_reason")
    return (
        float(best_val) if best_val is not None else None,
        float(test_val) if test_val is not None else None,
        int(best_epoch) if best_epoch is not None else None,
        str(stop_reason) if stop_reason is not None else None,
    )


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
            f"[skip] {spec.model}/{spec.run_id} "
            f"(best_val={best_val}, artifact={artifact_root.as_posix()})",
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
    print(f"\n[run] {spec.model}/{spec.run_id}", flush=True)
    print(f"  artifact_root={artifact_root.as_posix()}", flush=True)
    print(f"  cmd={' '.join(cmd)}", flush=True)

    if dry_run:
        return RunResult(
            spec=spec,
            artifact_root=artifact_root.as_posix(),
            status="ok",
            summary=None,
        )

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
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
        f"[done] {spec.model}/{spec.run_id} "
        f"best_val={best_val} test@best={test_val} epoch={best_epoch}",
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


def select_best_for_phase(results: list[RunResult], phase: str, model: str) -> Optional[RunSpec]:
    candidates = [
        r
        for r in results
        if r.spec.phase == phase
        and r.spec.model == model
        and r.status in ("ok", "skipped")
        and r.best_val_final_accuracy is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.best_val_final_accuracy or -1.0).spec


def select_best_overall(results: list[RunResult], model: str) -> Optional[RunResult]:
    candidates = [
        r
        for r in results
        if r.spec.model == model
        and r.status in ("ok", "skipped")
        and r.best_val_final_accuracy is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.best_val_final_accuracy or -1.0)


def load_grid_summary(grid_root: Path) -> Optional[dict[str, Any]]:
    path = grid_root / "grid_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def result_from_saved(entry: dict[str, Any]) -> RunResult:
    spec = RunSpec(
        model=str(entry["model"]),
        phase=str(entry["phase"]),
        run_id=str(entry["run_id"]),
        n_fft=int(entry["n_fft"]),
        hop_length=int(entry["hop_length"]),
        k=int(entry.get("k", 0)),
        p=int(entry.get("p", 1)),
        recurrency=int(entry.get("recurrency", 2)),
        branch_layers=int(entry.get("branch_layers", 2)),
        hidden_size=int(entry.get("hidden_size", 128)),
        subband_preset=str(entry.get("subband_preset", DEFAULT_SUBBAND_PRESET)),
    )
    return RunResult(
        spec=spec,
        artifact_root=str(entry.get("artifact_root", "")),
        status=str(entry.get("status", "ok")),
        best_val_final_accuracy=entry.get("best_val_final_accuracy"),
        test_final_accuracy_at_best_val=entry.get("test_final_accuracy_at_best_val"),
        best_epoch=entry.get("best_epoch"),
        stop_reason=entry.get("stop_reason"),
        error=entry.get("error"),
        summary=entry.get("summary"),
    )



def write_grid_summary(grid_root: Path, payload: dict[str, Any]) -> None:
    grid_root.mkdir(parents=True, exist_ok=True)
    path = grid_root / "grid_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[grid] updated {path.as_posix()}", flush=True)


def build_payload(
    grid_root: Path,
    all_results: list[RunResult],
    *,
    phase1_winners: dict[str, Optional[RunSpec]],
    phase2_winners: dict[str, Optional[RunSpec]],
    phase3_winners: dict[str, Optional[RunSpec]],
    in_progress_phase: Optional[str],
) -> dict[str, Any]:
    fl_best = select_best_overall(all_results, MODEL_BRANCHES)
    rnn_best = select_best_overall(all_results, MODEL_RNN)
    return {
        "grid_root": grid_root.as_posix(),
        "variant": "sumfusion",
        "common_train_args": list(COMMON_TRAIN_ARGS),
        "phase1_nfft_hop": [PHASE1_NFFT, PHASE1_HOP],
        "phase2_nfft_hop_grid": [list(pair) for pair in PHASE2_NFFT_HOP_GRID],
        "phase3_subband_presets": list(PHASE3_SUBBAND_PRESETS),
        "in_progress_phase": in_progress_phase,
        "runs": [r.to_dict() for r in all_results],
        "phase1_winners": {
            MODEL_BRANCHES: asdict(phase1_winners[MODEL_BRANCHES])
            if phase1_winners.get(MODEL_BRANCHES) is not None
            else None,
            MODEL_RNN: asdict(phase1_winners[MODEL_RNN])
            if phase1_winners.get(MODEL_RNN) is not None
            else None,
        },
        "phase2_winners": {
            MODEL_BRANCHES: asdict(phase2_winners[MODEL_BRANCHES])
            if phase2_winners.get(MODEL_BRANCHES) is not None
            else None,
            MODEL_RNN: asdict(phase2_winners[MODEL_RNN])
            if phase2_winners.get(MODEL_RNN) is not None
            else None,
        },
        "phase3_winners": {
            MODEL_BRANCHES: asdict(phase3_winners[MODEL_BRANCHES])
            if phase3_winners.get(MODEL_BRANCHES) is not None
            else None,
            MODEL_RNN: asdict(phase3_winners[MODEL_RNN])
            if phase3_winners.get(MODEL_RNN) is not None
            else None,
        },
        "best_overall": {
            MODEL_BRANCHES: fl_best.to_dict() if fl_best is not None else None,
            MODEL_RNN: rnn_best.to_dict() if rnn_best is not None else None,
        },
    }


def run_phase_queue(
    queue: list[RunSpec],
    *,
    phase_name: str,
    grid_root: Path,
    all_results: list[RunResult],
    done_ids: set[str],
    phase1_winners: dict[str, Optional[RunSpec]],
    phase2_winners: dict[str, Optional[RunSpec]],
    phase3_winners: dict[str, Optional[RunSpec]],
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    print(f"\n=== {phase_name} ({len(queue)} runs) ===", flush=True)
    for spec in queue:
        if spec.run_id in done_ids:
            print(f"[resume-skip] already recorded: {spec.model}/{spec.run_id}", flush=True)
            continue
        result = run_one(
            spec,
            grid_root,
            cache_dir=args.cache_dir,
            device=args.device,
            seed=args.seed,
            dry_run=dry_run,
            force=args.force,
        )
        all_results.append(result)
        done_ids.add(spec.run_id)
        if not dry_run:
            write_grid_summary(
                grid_root,
                build_payload(
                    grid_root,
                    all_results,
                    phase1_winners=phase1_winners,
                    phase2_winners=phase2_winners,
                    phase3_winners=phase3_winners,
                    in_progress_phase=phase_name,
                ),
            )


def main() -> int:
    args = parse_args()
    grid_root = Path(args.grid_root)
    dry_run = bool(args.dry_run)

    if not FAST_LEARNING_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {FAST_LEARNING_SCRIPT}")
    if not FAST_LEARNING_RNN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {FAST_LEARNING_RNN_SCRIPT}")

    saved = load_grid_summary(grid_root)
    all_results: list[RunResult] = []
    done_ids: set[str] = set()
    if saved is not None and isinstance(saved.get("runs"), list):
        for entry in saved["runs"]:
            if not isinstance(entry, dict):
                continue
            result = result_from_saved(entry)
            all_results.append(result)
            if result.status in ("ok", "skipped"):
                done_ids.add(result.spec.run_id)
        print(f"[resume] loaded {len(all_results)} prior runs from grid_summary.json", flush=True)

    phase1_winners: dict[str, Optional[RunSpec]] = {MODEL_BRANCHES: None, MODEL_RNN: None}
    phase2_winners: dict[str, Optional[RunSpec]] = {MODEL_BRANCHES: None, MODEL_RNN: None}
    phase3_winners: dict[str, Optional[RunSpec]] = {MODEL_BRANCHES: None, MODEL_RNN: None}

    phase1_queue = interleave_specs(build_branches_phase1_specs(), build_rnn_phase1_specs())
    run_phase_queue(
        phase1_queue,
        phase_name="phase1",
        grid_root=grid_root,
        all_results=all_results,
        done_ids=done_ids,
        phase1_winners=phase1_winners,
        phase2_winners=phase2_winners,
        phase3_winners=phase3_winners,
        args=args,
        dry_run=dry_run,
    )

    for model in (MODEL_BRANCHES, MODEL_RNN):
        winner = select_best_for_phase(all_results, "phase1", model)
        phase1_winners[model] = winner
        if winner is None:
            print(f"[warn] No successful phase1 runs for {model}; skipping later phases.", flush=True)
        else:
            print(
                f"[phase1 best] {model}: {winner.run_id} "
                f"(k={winner.k}, H={winner.hidden_size})",
                flush=True,
            )

    phase2_specs: list[RunSpec] = []
    if phase1_winners[MODEL_BRANCHES] is not None:
        phase2_specs.extend(build_phase2_specs(phase1_winners[MODEL_BRANCHES]))
    if phase1_winners[MODEL_RNN] is not None:
        phase2_specs.extend(build_phase2_specs(phase1_winners[MODEL_RNN]))
    branches_phase2 = [s for s in phase2_specs if s.model == MODEL_BRANCHES]
    rnn_phase2 = [s for s in phase2_specs if s.model == MODEL_RNN]
    run_phase_queue(
        interleave_specs(branches_phase2, rnn_phase2),
        phase_name="phase2",
        grid_root=grid_root,
        all_results=all_results,
        done_ids=done_ids,
        phase1_winners=phase1_winners,
        phase2_winners=phase2_winners,
        phase3_winners=phase3_winners,
        args=args,
        dry_run=dry_run,
    )

    for model in (MODEL_BRANCHES, MODEL_RNN):
        winner = select_best_for_phase(all_results, "phase2", model)
        phase2_winners[model] = winner
        if winner is None:
            print(f"[warn] No successful phase2 runs for {model}; skipping phase3.", flush=True)
        else:
            print(
                f"[phase2 best] {model}: {winner.run_id} "
                f"(n_fft={winner.n_fft}, hop={winner.hop_length})",
                flush=True,
            )

    phase3_specs: list[RunSpec] = []
    if phase2_winners[MODEL_BRANCHES] is not None:
        phase3_specs.extend(build_phase3_specs(phase2_winners[MODEL_BRANCHES]))
    if phase2_winners[MODEL_RNN] is not None:
        phase3_specs.extend(build_phase3_specs(phase2_winners[MODEL_RNN]))
    branches_phase3 = [s for s in phase3_specs if s.model == MODEL_BRANCHES]
    rnn_phase3 = [s for s in phase3_specs if s.model == MODEL_RNN]
    run_phase_queue(
        interleave_specs(branches_phase3, rnn_phase3),
        phase_name="phase3",
        grid_root=grid_root,
        all_results=all_results,
        done_ids=done_ids,
        phase1_winners=phase1_winners,
        phase2_winners=phase2_winners,
        phase3_winners=phase3_winners,
        args=args,
        dry_run=dry_run,
    )

    for model in (MODEL_BRANCHES, MODEL_RNN):
        winner = select_best_for_phase(all_results, "phase3", model)
        phase3_winners[model] = winner

    payload = build_payload(
        grid_root,
        all_results,
        phase1_winners=phase1_winners,
        phase2_winners=phase2_winners,
        phase3_winners=phase3_winners,
        in_progress_phase=None,
    )
    if not dry_run:
        write_grid_summary(grid_root, payload)

    print("\n=== Best overall ===", flush=True)
    for model in (MODEL_BRANCHES, MODEL_RNN):
        best = select_best_overall(all_results, model)
        if best is None:
            print(f"{model}: no successful runs", flush=True)
            continue
        s = best.spec
        extra = (
            f"L={s.branch_layers}" if model == MODEL_BRANCHES else f"R={s.recurrency}"
        )
        print(
            f"{model}: {s.run_id} | val={best.best_val_final_accuracy:.4f} "
            f"test={best.test_final_accuracy_at_best_val} | "
            f"n_fft={s.n_fft} hop={s.hop_length} k={s.k} {extra} H={s.hidden_size} "
            f"preset={s.subband_preset}",
            flush=True,
        )

    failed = [r for r in all_results if r.status == "failed"]
    if failed:
        print(f"\n[warn] {len(failed)} run(s) failed.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
