#!/usr/bin/env python
"""Per-frame accuracy and ED-sKWS early-exit threshold sweep.

Inference only. Loads existing checkpoints, runs val/test, and writes:

- accuracy of ``argmax(z_t)`` at every frame (our decision);
- accuracy of ``argmax(O_t)`` where ``O_t = cumsum softmax(z_i)``;
- early exit when ``max(softmax(O_t)) > C`` for ``C = 0.05, 0.10, ..., 0.95``.

The second softmax does not change the class. It only puts the cumulative
vote on a common (0, 1) scale so one threshold works at every frame.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SNK_DIR = REPO_ROOT / "snn_kws"
if str(SNK_DIR) not in sys.path:
    sys.path.insert(0, str(SNK_DIR))

# ED-sKWS Table 1, 512-unit model: mean stop 60.46 of 98 frames.
ED_SKWS_STOP_FRAME = 60.46
ED_SKWS_N_FRAMES = 98
ED_SKWS_TIME_MS = ED_SKWS_STOP_FRAME / ED_SKWS_N_FRAMES * 1000.0
ED_SKWS_EARLY_ACC = 0.9304
ED_SKWS_LATE_ACC = 0.9315

DEFAULT_GLOBS = (
    "results/experiments/table2_multiseed/runs/rnn_k2_R3_H256_nfft512_hop64_p3_default_gsu_seed*",
    "results/experiments/table2_multiseed/runs/ff_k2_L2_H256_nfft512_hop64_p3_default_gsu_seed*",
)
ACCURACY_TOLERANCE = 0.002

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.8,
        "legend.framealpha": 0.85,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
    }
)


def thresholds_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 1.0, 0.05), 2)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def mean_std_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def _filter_config(config_cls, config_dict: dict) -> dict:
    known = {field.name for field in dataclasses.fields(config_cls)}
    return {key: value for key, value in config_dict.items() if key in known}


def discover_run_dirs(globs: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in globs:
        matches = sorted(REPO_ROOT.glob(pattern))
        found.extend(path for path in matches if path.is_dir())
    # Unique, seed 7 first within a config so the sanity check shows up early.
    unique = list(dict.fromkeys(found))

    def sort_key(path: Path) -> tuple:
        match = re.search(r"_seed(\d+)$", path.name)
        seed = int(match.group(1)) if match else 10**9
        arch_order = 0 if path.name.startswith("rnn_") else 1
        return (arch_order, path.name.rsplit("_seed", 1)[0], seed)

    return sorted(unique, key=sort_key)


def parse_run_identity(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    match = re.match(r"(rnn|ff)_k(\d+)_(R|L)(\d+)_", name)
    seed_match = re.search(r"_seed(\d+)$", name)
    if match is None or seed_match is None:
        raise ValueError(f"Cannot parse architecture/seed from run dir {name}")
    arch = match.group(1)
    return {
        "run_id": name,
        "config_key": name.rsplit("_seed", 1)[0],
        "arch": arch,
        "k": int(match.group(2)),
        "depth": int(match.group(4)),
        "depth_tag": match.group(3),
        "seed": int(seed_match.group(1)),
        "label": ("RNN" if arch == "rnn" else "GSN-layers")
        + f" k={match.group(2)} {match.group(3)}={match.group(4)}",
    }


def find_pkl(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one .pkl in {run_dir}, found {len(matches)}")
    return matches[0]


def expected_accuracy(summary: dict, split: str) -> float:
    inner = summary.get("summary") or {}
    key = "test_final_accuracy_at_best_val" if split == "test" else "best_val_final_accuracy"
    value = inner.get(key)
    if value is None:
        raise KeyError(f"{key} missing from best_summary.json")
    return float(value)


def load_model(ckpt_path: Path, cache_dir: Path, device: torch.device):
    with ckpt_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    config_dict = dict(checkpoint["config"])
    if "recurrency" in checkpoint or "recurrency" in config_dict:
        import train_rnn as train_mod

        if "recurrency" not in config_dict:
            config_dict["recurrency"] = int(checkpoint.get("recurrency", 2))
        config_cls = train_mod.SpikeFusionConfig
        cfg = config_cls(**_filter_config(config_cls, config_dict))
        cfg.cache_dir = str(cache_dir)
        cfg.precompute_in_memory = True
        model = train_mod.StreamingSpikeFusionClassifier(config=cfg, k=int(checkpoint["k"]))
    elif "p" in checkpoint:
        import train_layered as train_mod

        config_cls = train_mod.SpikeFusionConfig
        cfg = config_cls(**_filter_config(config_cls, config_dict))
        cfg.cache_dir = str(cache_dir)
        cfg.precompute_in_memory = True
        model = train_mod.StreamingSpikeFusionClassifier(
            config=cfg,
            k=int(checkpoint["k"]),
            p=int(checkpoint["p"]),
        )
    else:
        raise ValueError(f"Unrecognized checkpoint layout: {ckpt_path}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, train_mod


def make_split_loader(train_mod, config, split: str, batch_size: int, max_batches: Optional[int]):
    rows = train_mod.select_gsc_rows(config, split=split)
    if max_batches is not None:
        rows = rows[: max(1, int(max_batches) * int(batch_size))]
    dataset = train_mod.StreamingGSCDataset(rows, config, precompute_in_memory=True)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=train_mod.streaming_gsc_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


@torch.inference_mode()
def evaluate_loader(model, loader, thresholds: np.ndarray, device: torch.device) -> dict[str, Any]:
    thresholds_t = torch.as_tensor(thresholds, device=device, dtype=torch.float32)
    n_thr = int(thresholds_t.numel())
    correct_z: Optional[torch.Tensor] = None
    correct_o: Optional[torch.Tensor] = None
    correct_stop = torch.zeros(n_thr, device=device, dtype=torch.float64)
    n_frames: Optional[int] = None
    total = 0
    stop_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(frames, return_dynamics=False)["decoder_logits"]
        batch_size, steps, _ = logits.shape
        if n_frames is None:
            n_frames = int(steps)
            correct_z = torch.zeros(steps, device=device, dtype=torch.float64)
            correct_o = torch.zeros(steps, device=device, dtype=torch.float64)
        elif steps != n_frames:
            raise RuntimeError(f"Frame count changed from {n_frames} to {steps}")

        probs = torch.softmax(logits, dim=-1)
        votes = torch.cumsum(probs, dim=1)
        pred_z = logits.argmax(dim=-1)
        pred_o = votes.argmax(dim=-1)
        label_cols = labels.unsqueeze(1)
        correct_z += (pred_z == label_cols).sum(dim=0)
        correct_o += (pred_o == label_cols).sum(dim=0)

        confidence = torch.softmax(votes, dim=-1).amax(dim=-1)
        exceeded = confidence.unsqueeze(1) > thresholds_t.view(1, n_thr, 1)
        has_stop = exceeded.any(dim=-1)
        first = exceeded.to(dtype=torch.int64).argmax(dim=-1)
        stop = torch.where(has_stop, first, steps - 1)
        chosen = pred_o.gather(1, stop)
        correct_stop += (chosen == labels.unsqueeze(1)).sum(dim=0)
        stop_chunks.append(stop.cpu())
        total += int(batch_size)
        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == n_batches:
            elapsed = time.perf_counter() - started
            print(
                f"  batch {batch_idx}/{n_batches} examples={total} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if n_frames is None or correct_z is None or correct_o is None or total == 0:
        raise RuntimeError("Loader produced no examples")

    stop_all = torch.cat(stop_chunks, dim=0).numpy()
    denom = float(total)
    return {
        "n_examples": int(total),
        "n_frames": int(n_frames),
        "accuracy_logits": (correct_z / denom).cpu().numpy(),
        "accuracy_cumulative": (correct_o / denom).cpu().numpy(),
        "stop_frames": stop_all,
        "threshold_accuracy": (correct_stop / denom).cpu().numpy(),
    }


def frame_time_ms(n_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) * float(hop_length) / float(sample_rate) * 1000.0


def nearest_frame(time_ms: np.ndarray, target_ms: float) -> int:
    return int(np.argmin(np.abs(time_ms - target_ms)))


def threshold_rows(
    thresholds: np.ndarray,
    stop_frames: np.ndarray,
    threshold_accuracy: np.ndarray,
    hop_length: int,
    sample_rate: int,
    n_frames: int,
) -> list[dict[str, float]]:
    rows = []
    last = n_frames - 1
    for index, threshold in enumerate(thresholds):
        stops = stop_frames[:, index].astype(np.float64)
        mean_frame = float(stops.mean())
        median_frame = float(np.median(stops))
        ms_per_frame = float(hop_length) / float(sample_rate) * 1000.0
        rows.append(
            {
                "c": float(threshold),
                "accuracy": float(threshold_accuracy[index]),
                "mean_stop_frame": mean_frame,
                "median_stop_frame": median_frame,
                "mean_stop_ms": mean_frame * ms_per_frame,
                "median_stop_ms": median_frame * ms_per_frame,
                "fraction_early": float(np.mean(stops < last)),
            }
        )
    return rows


def evaluate_run(
    run_dir: Path,
    *,
    splits: list[str],
    batch_size: int,
    device: torch.device,
    cache_dir: Path,
    out_dir: Path,
    thresholds: np.ndarray,
    max_batches: Optional[int],
    force: bool,
) -> list[dict[str, Any]]:
    identity = parse_run_identity(run_dir)
    summary_path = run_dir / "best_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt_path = find_pkl(run_dir)
    written: list[dict[str, Any]] = []
    model = None
    train_mod = None
    config = None

    for split in splits:
        out_path = out_dir / "runs" / f"{identity['run_id']}_{split}.json"
        if out_path.is_file() and not force and max_batches is None:
            print(f"[skip] {identity['run_id']} {split} ({out_path.name})", flush=True)
            written.append(json.loads(out_path.read_text(encoding="utf-8")))
            continue
        if model is None:
            print(f"[load] {identity['run_id']} {ckpt_path.name}", flush=True)
            model, config, train_mod = load_model(ckpt_path, cache_dir, device)
        assert config is not None and train_mod is not None and model is not None
        print(f"[start] {identity['run_id']} {split}", flush=True)
        started = time.perf_counter()
        loader = make_split_loader(train_mod, config, split, batch_size, max_batches)
        metrics = evaluate_loader(model, loader, thresholds, device)
        del loader
        time_ms = frame_time_ms(metrics["n_frames"], int(config.hop_length), int(config.sample_rate))
        final_acc = float(metrics["accuracy_logits"][-1])
        expected = expected_accuracy(summary, split)
        delta = abs(final_acc - expected)
        if max_batches is None and delta > ACCURACY_TOLERANCE:
            raise RuntimeError(
                f"{identity['run_id']} {split}: final accuracy {final_acc:.6f} "
                f"disagrees with best_summary {expected:.6f} (delta={delta:.6f})"
            )
        ref_idx = nearest_frame(time_ms, ED_SKWS_TIME_MS)
        payload = {
            **identity,
            "split": split,
            "checkpoint": str(ckpt_path),
            "n_examples": metrics["n_examples"],
            "n_frames": metrics["n_frames"],
            "hop_length": int(config.hop_length),
            "sample_rate": int(config.sample_rate),
            "frame_time_ms": time_ms.tolist(),
            "accuracy_logits": metrics["accuracy_logits"].tolist(),
            "accuracy_cumulative": metrics["accuracy_cumulative"].tolist(),
            "final_accuracy_logits": final_acc,
            "final_accuracy_cumulative": float(metrics["accuracy_cumulative"][-1]),
            "expected_final_accuracy": expected,
            "final_accuracy_delta": delta,
            "ed_skws_time_ms": ED_SKWS_TIME_MS,
            "accuracy_logits_at_ed_skws_time": float(metrics["accuracy_logits"][ref_idx]),
            "accuracy_cumulative_at_ed_skws_time": float(metrics["accuracy_cumulative"][ref_idx]),
            "frame_at_ed_skws_time": ref_idx,
            "thresholds": threshold_rows(
                thresholds,
                metrics["stop_frames"],
                metrics["threshold_accuracy"],
                int(config.hop_length),
                int(config.sample_rate),
                metrics["n_frames"],
            ),
            "elapsed_sec": time.perf_counter() - started,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[done] {identity['run_id']} {split} final={final_acc:.4f} "
            f"expected={expected:.4f} delta={delta:.6f} "
            f"at_{ED_SKWS_TIME_MS:.0f}ms={payload['accuracy_logits_at_ed_skws_time']:.4f} "
            f"({payload['elapsed_sec']:.1f}s)",
            flush=True,
        )
        written.append(payload)

    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return written


def _group_key(row: dict) -> tuple[str, str]:
    return (row["config_key"], row["split"])


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    aggregated = []
    for (config_key, split), members in sorted(groups.items()):
        members = sorted(members, key=lambda item: int(item["seed"]))
        time_ms = np.asarray(members[0]["frame_time_ms"], dtype=np.float64)
        logits = np.stack([np.asarray(item["accuracy_logits"]) for item in members])
        cumulative = np.stack([np.asarray(item["accuracy_cumulative"]) for item in members])
        if any(item["n_frames"] != members[0]["n_frames"] for item in members):
            raise RuntimeError(f"Frame count mismatch inside {config_key} {split}")
        logits_mean, logits_std = mean_std_stack(logits)
        cum_mean, cum_std = mean_std_stack(cumulative)
        ref_idx = nearest_frame(time_ms, ED_SKWS_TIME_MS)
        threshold_values = [item["c"] for item in members[0]["thresholds"]]
        threshold_summary = []
        for index, threshold in enumerate(threshold_values):
            acc_mean, acc_std = mean_std([item["thresholds"][index]["accuracy"] for item in members])
            stop_mean, stop_std = mean_std([item["thresholds"][index]["mean_stop_ms"] for item in members])
            med_mean, med_std = mean_std([item["thresholds"][index]["median_stop_ms"] for item in members])
            early_mean, early_std = mean_std(
                [item["thresholds"][index]["fraction_early"] for item in members]
            )
            threshold_summary.append(
                {
                    "c": float(threshold),
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "mean_stop_ms_mean": stop_mean,
                    "mean_stop_ms_std": stop_std,
                    "median_stop_ms_mean": med_mean,
                    "median_stop_ms_std": med_std,
                    "fraction_early_mean": early_mean,
                    "fraction_early_std": early_std,
                }
            )
        at_time_mean, at_time_std = mean_std(
            [item["accuracy_logits_at_ed_skws_time"] for item in members]
        )
        at_time_cum_mean, at_time_cum_std = mean_std(
            [item["accuracy_cumulative_at_ed_skws_time"] for item in members]
        )
        final_mean, final_std = mean_std([item["final_accuracy_logits"] for item in members])
        aggregated.append(
            {
                "config_key": config_key,
                "label": members[0]["label"],
                "arch": members[0]["arch"],
                "split": split,
                "n_seeds": len(members),
                "seeds": [int(item["seed"]) for item in members],
                "frame_time_ms": time_ms.tolist(),
                "accuracy_logits_mean": logits_mean.tolist(),
                "accuracy_logits_std": logits_std.tolist(),
                "accuracy_cumulative_mean": cum_mean.tolist(),
                "accuracy_cumulative_std": cum_std.tolist(),
                "final_accuracy_mean": final_mean,
                "final_accuracy_std": final_std,
                "frame_at_ed_skws_time": ref_idx,
                "accuracy_logits_at_ed_skws_time_mean": at_time_mean,
                "accuracy_logits_at_ed_skws_time_std": at_time_std,
                "accuracy_cumulative_at_ed_skws_time_mean": at_time_cum_mean,
                "accuracy_cumulative_at_ed_skws_time_std": at_time_cum_std,
                "thresholds": threshold_summary,
            }
        )
    return {
        "ed_skws_reference": {
            "stop_frame_mean": ED_SKWS_STOP_FRAME,
            "n_frames": ED_SKWS_N_FRAMES,
            "time_ms": ED_SKWS_TIME_MS,
            "early_accuracy": ED_SKWS_EARLY_ACC,
            "late_accuracy": ED_SKWS_LATE_ACC,
            "note": (
                "Mean decision time on GSC, not a fixed frame index. "
                "Their hop is about 10 ms; ours is 4 ms."
            ),
        },
        "groups": aggregated,
    }


def _plot_band(ax, x, mean, std, color, label, linestyle):
    mean = np.asarray(mean)
    std = np.asarray(std)
    ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)


def plot_accuracy(groups: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {"rnn": "C0", "ff": "C1"}
    for group in groups:
        color = colors.get(group["arch"], "C2")
        x = np.asarray(group["frame_time_ms"])
        _plot_band(
            ax,
            x,
            group["accuracy_logits_mean"],
            group["accuracy_logits_std"],
            color,
            f"{group['label']}, argmax логита",
            "-",
        )
        _plot_band(
            ax,
            x,
            group["accuracy_cumulative_mean"],
            group["accuracy_cumulative_std"],
            color,
            f"{group['label']}, накопленный голос",
            "--",
        )
    ax.axvline(ED_SKWS_TIME_MS, color="0.35", linestyle=":", linewidth=1.2)
    ax.text(
        ED_SKWS_TIME_MS + 8,
        0.08,
        f"ED-sKWS\n{ED_SKWS_STOP_FRAME:.1f}/{ED_SKWS_N_FRAMES}",
        color="0.25",
        fontsize=8,
        va="bottom",
    )
    ax.set_xlim(0, max(np.asarray(groups[0]["frame_time_ms"])[-1], ED_SKWS_TIME_MS))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Время решения, мс")
    ax.set_ylabel("Точность")
    ax.set_title("Точность, если решение принимается в этот момент (test, mean±std)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_thresholds(groups: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    colors = {"rnn": "C0", "ff": "C1"}
    for group in groups:
        color = colors.get(group["arch"], "C2")
        xs = [row["c"] for row in group["thresholds"]]
        acc = np.asarray([row["accuracy_mean"] for row in group["thresholds"]])
        acc_std = np.asarray([row["accuracy_std"] for row in group["thresholds"]])
        stop = np.asarray([row["mean_stop_ms_mean"] for row in group["thresholds"]])
        stop_std = np.asarray([row["mean_stop_ms_std"] for row in group["thresholds"]])
        _plot_band(axes[0], xs, acc, acc_std, color, group["label"], "-")
        _plot_band(axes[1], xs, stop, stop_std, color, group["label"], "-")
    axes[0].axhline(ED_SKWS_EARLY_ACC, color="0.35", linestyle=":", linewidth=1.2)
    axes[0].text(0.06, ED_SKWS_EARLY_ACC - 0.04, "ED-sKWS early 93.04%", color="0.25", fontsize=8)
    axes[0].set_xlabel("Порог уверенности C")
    axes[0].set_ylabel("Точность раннего выхода")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Точность")
    axes[0].legend(loc="lower right")
    axes[1].axhline(ED_SKWS_TIME_MS, color="0.35", linestyle=":", linewidth=1.2)
    axes[1].text(0.06, ED_SKWS_TIME_MS + 15, "ED-sKWS ~617 мс", color="0.25", fontsize=8)
    axes[1].set_xlabel("Порог уверенности C")
    axes[1].set_ylabel("Среднее время стопа, мс")
    axes[1].set_title("Когда срабатывает порог")
    axes[1].legend(loc="upper left")
    fig.suptitle("Ранний выход: max softmax(cumsum softmax(z)) > C (test, mean±std)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-frame accuracy and ED-sKWS threshold sweep")
    parser.add_argument("--run-glob", action="append", default=None, help="Run-dir glob, repeatable.")
    parser.add_argument("--splits", nargs="+", default=["test", "val"], choices=("test", "val"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "google_speech_commands")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "experiments" / "early_decision")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug cap. Skips the accuracy check.")
    parser.add_argument("--force", action="store_true", help="Recompute even if the run JSON exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    globs = args.run_glob or list(DEFAULT_GLOBS)
    run_dirs = discover_run_dirs(globs)
    if not run_dirs:
        raise SystemExit(f"No run directories matched {globs}")
    device = torch.device(args.device)
    thresholds = thresholds_grid()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    print(f"Runs: {len(run_dirs)}", flush=True)
    for run_dir in run_dirs:
        print(f"  {run_dir.name}", flush=True)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(
            evaluate_run(
                run_dir,
                splits=list(args.splits),
                batch_size=int(args.batch_size),
                device=device,
                cache_dir=args.cache_dir,
                out_dir=out_dir,
                thresholds=thresholds,
                max_batches=args.max_batches,
                force=bool(args.force),
            )
        )

    summary = aggregate(rows)
    summary_path = out_dir / "aggregate.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    test_groups = [group for group in summary["groups"] if group["split"] == "test"]
    if test_groups:
        plot_accuracy(test_groups, out_dir / "accuracy_vs_time_test.png")
        plot_thresholds(test_groups, out_dir / "threshold_sweep_test.png")
    print(f"[wrote] {summary_path}", flush=True)
    for group in summary["groups"]:
        print(
            f"[agg] {group['label']} {group['split']} seeds={group['seeds']} "
            f"final={group['final_accuracy_mean']:.4f}±{group['final_accuracy_std']:.4f} "
            f"at_{ED_SKWS_TIME_MS:.0f}ms="
            f"{group['accuracy_logits_at_ed_skws_time_mean']:.4f}"
            f"±{group['accuracy_logits_at_ed_skws_time_std']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
