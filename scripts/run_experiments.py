#!/usr/bin/env python3
"""Multi-seed experiment runner for the additional paper experiments.

Plans
-----
``table2``
    Full architecture grid for the updated Table 2: k in {0,1,2} x depth in
    {2,3,4} (L for GSU_ff / R for GSU_rnn), H=256, n_fft=512, hop=64,
    preset p3_default, neuron gsu. Every config is trained with each seed from
    ``--seeds`` (fixed list per supervisor decision). The training scripts
    themselves add ``num_params`` and the validation spike-count statistics to
    each run's ``best_summary.json``.

``hidden_search``
    New Table 3: H in {32, 64, 128} for the best config of each architecture.
    Best configs are taken from the table2 aggregate (highest mean val
    accuracy across seeds) or overridden via ``--ff-best`` / ``--rnn-best``.

Storage layout
--------------
    <exp_root>/runs/<run_id>/        history.json, best_summary.json, train.log, checkpoint
    <exp_root>/aggregate.json        per config: per-seed metrics + mean/std across seeds
    <exp_root>/aggregate.csv         same, flat table (for the paper / supervisor)

The aggregate is rewritten after every finished run, so the script is
crash-resumable: existing runs with a best_summary.json are skipped
(use --force to retrain).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from grid_search import COMMON_TRAIN_ARGS  # noqa: E402

TRAIN_SCRIPTS = {
    "ff": REPO_ROOT / "snn_kws" / "train_layered.py",
    "rnn": REPO_ROOT / "snn_kws" / "train_rnn.py",
}

DEFAULT_SEEDS = (7, 10, 12, 80)
DEFAULT_EXPERIMENT_ROOTS = {
    "table2": REPO_ROOT / "results" / "experiments" / "table2_multiseed",
    "hidden_search": REPO_ROOT / "results" / "experiments" / "hidden_search",
}

# Published best configs (Table 2 of the manuscript); used as fallback for
# hidden_search when no table2 aggregate is available yet.
FALLBACK_BEST = {"ff": (2, 2), "rnn": (2, 3)}  # arch -> (k, depth)

HIDDEN_SEARCH_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class RunSpec:
    arch: str  # "ff" | "rnn"
    k: int
    depth: int  # L for ff, R for rnn
    hidden_size: int = 256
    n_fft: int = 512
    hop_length: int = 64
    subband_preset: str = "p3_default"
    neuron_type: str = "gsu"
    seed: int = 7

    @property
    def config_key(self) -> str:
        """Identifies the config regardless of seed (aggregation group)."""
        depth_tag = "L" if self.arch == "ff" else "R"
        return (
            f"{self.arch}_k{self.k}_{depth_tag}{self.depth}_H{self.hidden_size}"
            f"_nfft{self.n_fft}_hop{self.hop_length}_{self.subband_preset}"
            f"_{self.neuron_type}"
        )

    @property
    def run_id(self) -> str:
        return f"{self.config_key}_seed{self.seed}"

    def config_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "k": self.k,
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "subband_preset": self.subband_preset,
            "neuron_type": self.neuron_type,
        }


def build_command(spec: RunSpec, run_dir: Path, *, cache_dir: str, device: Optional[str]) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(TRAIN_SCRIPTS[spec.arch]),
        "--artifact-root",
        str(run_dir),
        "--log-file",
        str(run_dir / "train.log"),
        "--cache-dir",
        cache_dir,
        "--seed",
        str(spec.seed),
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
        "--neuron-type",
        spec.neuron_type,
        *COMMON_TRAIN_ARGS,
    ]
    if spec.arch == "ff":
        cmd.extend(["--p", "1", "--branch-layers", str(spec.depth)])
    else:
        cmd.extend(["--recurrency", str(spec.depth)])
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def load_best_summary(run_dir: Path) -> Optional[dict[str, Any]]:
    path = run_dir / "best_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def extract_run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    inner = summary.get("summary") or {}
    keys = (
        "best_val_final_accuracy",
        "test_final_accuracy_at_best_val",
        "num_params",
        "val_spikes_per_example_mean",
        "val_spikes_per_example_std",
        "best_epoch",
        "stop_reason",
    )
    return {key: inner.get(key) for key in keys}


def mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var**0.5


def aggregate_experiment(exp_root: Path) -> dict[str, Any]:
    """Scan runs/, group by config, compute mean/std across seeds."""
    runs_dir = exp_root / "runs"
    groups: dict[str, dict[str, Any]] = {}
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary = load_best_summary(run_dir)
            if summary is None:
                continue
            config = summary.get("config") or {}
            inner = summary.get("summary") or {}
            # Rebuild the grouping key from the stored config; fall back to the
            # run directory name minus the seed suffix.
            try:
                arch = "rnn" if "recurrency" in config else "ff"
                depth = config.get("recurrency") if arch == "rnn" else config.get("branch_layers")
                spec = RunSpec(
                    arch=arch,
                    k=int(config.get("k", inner.get("k"))),
                    depth=int(depth),
                    hidden_size=int(config.get("hidden_size", 256)),
                    n_fft=int(config.get("n_fft", 512)),
                    hop_length=int(config.get("hop_length", 64)),
                    subband_preset=str(config.get("subband_preset", "p3_default")),
                    neuron_type=str(config.get("neuron_type", "gsu")),
                    seed=int(config.get("seed", inner.get("seed", 0))),
                )
                key = spec.config_key
                config_info = spec.config_dict()
            except (TypeError, ValueError):
                key = run_dir.name.rsplit("_seed", 1)[0]
                config_info = {}
            metrics = extract_run_metrics(summary)
            metrics["seed"] = int(config.get("seed", inner.get("seed", -1)))
            metrics["run_id"] = run_dir.name
            group = groups.setdefault(key, {"config": config_info, "runs": []})
            group["runs"].append(metrics)

    configs_out = []
    for key in sorted(groups):
        group = groups[key]
        runs = group["runs"]
        val_mean, val_std = mean_std([r["best_val_final_accuracy"] for r in runs])
        test_mean, test_std = mean_std([r["test_final_accuracy_at_best_val"] for r in runs])
        spikes_mean, spikes_seed_std = mean_std(
            [r["val_spikes_per_example_mean"] for r in runs]
        )
        within_std, _ = mean_std([r["val_spikes_per_example_std"] for r in runs])
        num_params = next(
            (r["num_params"] for r in runs if r["num_params"] is not None), None
        )
        configs_out.append(
            {
                "config_key": key,
                **group["config"],
                "n_seeds": len(runs),
                "seeds": [r["seed"] for r in runs],
                "val_final_accuracy_mean": val_mean,
                "val_final_accuracy_std": val_std,
                "test_final_accuracy_mean": test_mean,
                "test_final_accuracy_std": test_std,
                "num_params": num_params,
                "val_spikes_per_example_mean": spikes_mean,
                "val_spikes_per_example_std_across_seeds": spikes_seed_std,
                "val_spikes_per_example_std_within": within_std,
                "runs": runs,
            }
        )

    payload = {"experiment_root": exp_root.as_posix(), "configs": configs_out}
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "aggregate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    csv_columns = (
        "config_key", "arch", "k", "depth", "hidden_size", "n_fft", "hop_length",
        "subband_preset", "neuron_type", "n_seeds",
        "val_final_accuracy_mean", "val_final_accuracy_std",
        "test_final_accuracy_mean", "test_final_accuracy_std",
        "num_params",
        "val_spikes_per_example_mean", "val_spikes_per_example_std_across_seeds",
        "val_spikes_per_example_std_within",
    )
    with (exp_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for row in configs_out:
            writer.writerow(row)
    print(f"[aggregate] updated {(exp_root / 'aggregate.json').as_posix()}", flush=True)
    return payload


def pick_best_configs(table2_aggregate: Path) -> dict[str, tuple[int, int]]:
    """Best (k, depth) per arch by mean val accuracy from a table2 aggregate."""
    payload = json.loads(table2_aggregate.read_text(encoding="utf-8"))
    best: dict[str, tuple[int, int]] = {}
    best_score: dict[str, float] = {}
    for entry in payload.get("configs", []):
        arch = entry.get("arch")
        score = entry.get("val_final_accuracy_mean")
        if arch not in ("ff", "rnn") or score is None:
            continue
        if arch not in best or score > best_score[arch]:
            best[arch] = (int(entry["k"]), int(entry["depth"]))
            best_score[arch] = float(score)
    return best


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if args.experiment == "table2":
        for arch in args.archs:
            for k in args.ks:
                for depth in args.depths:
                    for seed in args.seeds:
                        specs.append(
                            RunSpec(arch=arch, k=k, depth=depth, seed=int(seed))
                        )
    elif args.experiment == "hidden_search":
        best: dict[str, tuple[int, int]] = {}
        if args.table2_aggregate and Path(args.table2_aggregate).exists():
            best = pick_best_configs(Path(args.table2_aggregate))
            print(f"[plan] best configs from {args.table2_aggregate}: {best}", flush=True)
        for arch, override in (("ff", args.ff_best), ("rnn", args.rnn_best)):
            if override is not None:
                k_str, d_str = override.split(",")
                best[arch] = (int(k_str), int(d_str))
            elif arch not in best:
                best[arch] = FALLBACK_BEST[arch]
                print(
                    f"[plan] no aggregate for {arch}; using published best "
                    f"k={best[arch][0]}, depth={best[arch][1]}",
                    flush=True,
                )
        for arch in ("ff", "rnn"):
            k, depth = best[arch]
            for hidden in args.hidden_sizes:
                for seed in args.seeds:
                    specs.append(
                        RunSpec(
                            arch=arch,
                            k=k,
                            depth=depth,
                            hidden_size=int(hidden),
                            seed=int(seed),
                        )
                    )
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.experiment)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, choices=sorted(DEFAULT_EXPERIMENT_ROOTS))
    parser.add_argument(
        "--exp-root",
        default=None,
        help="Override the experiment root (default: results/experiments/<experiment>).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--archs",
        nargs="+",
        choices=("ff", "rnn"),
        default=["ff", "rnn"],
        help="(table2) restrict architectures, e.g. --archs rnn.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        choices=(0, 1, 2),
        default=[0, 1, 2],
        help="(table2) restrict k values, e.g. --ks 0.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        choices=(2, 3, 4),
        default=[2, 3, 4],
        help="(table2) restrict depths (L/R), e.g. --depths 3 4.",
    )
    parser.add_argument("--cache-dir", default="data/google_speech_commands")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain even if best_summary.json exists.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild aggregate.json/csv from existing runs.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=list(HIDDEN_SEARCH_SIZES),
        help="(hidden_search) hidden sizes to evaluate.",
    )
    parser.add_argument(
        "--table2-aggregate",
        default=None,
        help="(hidden_search) path to the table2 aggregate.json used to pick "
        "the best config per arch (default: <table2 root>/aggregate.json).",
    )
    parser.add_argument(
        "--ff-best",
        default=None,
        metavar="K,L",
        help="(hidden_search) override best ff config, e.g. '2,2'.",
    )
    parser.add_argument(
        "--rnn-best",
        default=None,
        metavar="K,R",
        help="(hidden_search) override best rnn config, e.g. '2,3'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_root = Path(args.exp_root) if args.exp_root else DEFAULT_EXPERIMENT_ROOTS[args.experiment]
    if args.table2_aggregate is None:
        args.table2_aggregate = str(DEFAULT_EXPERIMENT_ROOTS["table2"] / "aggregate.json")

    if args.aggregate_only:
        aggregate_experiment(exp_root)
        return 0

    specs = build_specs(args)
    print(f"[plan] experiment={args.experiment}: {len(specs)} runs "
          f"(seeds={list(args.seeds)})", flush=True)

    failed: list[str] = []
    for spec in specs:
        run_dir = exp_root / "runs" / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = load_best_summary(run_dir)
        if existing is not None and not args.force:
            metrics = extract_run_metrics(existing)
            print(
                f"[skip] {spec.run_id} (best_val={metrics['best_val_final_accuracy']})",
                flush=True,
            )
            continue
        cmd = build_command(spec, run_dir, cache_dir=args.cache_dir, device=args.device)
        print(f"\n[run] {spec.run_id}\n  cmd={' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            print(f"[fail] {spec.run_id} (exit {completed.returncode})", flush=True)
            failed.append(spec.run_id)
        else:
            metrics = extract_run_metrics(load_best_summary(run_dir) or {})
            print(
                f"[done] {spec.run_id} best_val={metrics['best_val_final_accuracy']} "
                f"test@best={metrics['test_final_accuracy_at_best_val']} "
                f"spikes={metrics['val_spikes_per_example_mean']}",
                flush=True,
            )
        aggregate_experiment(exp_root)

    if not args.dry_run:
        aggregate_experiment(exp_root)
    if failed:
        print(f"\n[warn] {len(failed)} run(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
