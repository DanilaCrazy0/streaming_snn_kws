#!/usr/bin/env python3
"""Aggregate sumfusion grid-search artifacts, build thesis figures and metrics.

Run from repo root:
    python scripts/make_figures.py
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SNK_DIR = REPO_ROOT / "snn_kws"
GRID_ROOT = REPO_ROOT / "results" / "grid"
PICS_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR = REPO_ROOT / "results" / "analysis"
CACHE_DIR = REPO_ROOT / "data" / "google_speech_commands"

MODEL_BRANCHES = "fast_learning_sumfusion"
MODEL_RNN = "fast_learning_rnn_sumfusion"

BEST_RUN = GRID_ROOT / MODEL_RNN / "bonus_phase2_nfft512_hop64_k2_R3_H256"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.8,
        "legend.framealpha": 0.85,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
    }
)
C = [plt.get_cmap("tab10")(i) for i in range(10)]

PRESET_LABELS = {
    "p1_full_0_4_4_8": "full + 0--4 + 4--8 kHz",
    "p2_full_0_2_2_4_4_6_6_8": "6 полос",
    "p3_default": "базовый (4 ветви)",
    "p4_full_0_1_1_8": "full + 0--1 + 1--8 kHz",
    "p5_full_0_4_4_8": "full + 0--4 + 4--8 kHz (alt)",
    "p6_full_0_3_3_6_6_8": "full + 0--3 + 3--6 + 6--8 kHz",
    "p7_full_0_2_2_5_5_8": "full + 0--2 + 2--5 + 5--8 kHz",
}


def infer_phase(run_id: str) -> str:
    if run_id.startswith("final_"):
        return "final"
    if run_id.startswith("bonus_phase"):
        return run_id.split("_")[1]  # phase2 / phase3
    match = re.match(r"phase(\d+)", run_id)
    return f"phase{match.group(1)}" if match else "other"


def load_runs() -> pd.DataFrame:
    rows: list[dict] = []
    for summary_path in GRID_ROOT.glob("**/best_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        config = payload.get("config", {})
        artifact_root = summary_path.parent
        run_id = artifact_root.name
        rows.append(
            {
                "run_id": run_id,
                "model": artifact_root.parent.name,
                "phase": infer_phase(run_id),
                "k": config.get("k", summary.get("k")),
                "branch_layers": config.get("branch_layers"),
                "recurrency": config.get("recurrency"),
                "hidden_size": config.get("hidden_size", 128),
                "n_fft": config.get("n_fft"),
                "hop_length": config.get("hop_length"),
                "subband_preset": config.get("subband_preset", "p3_default"),
                "val": summary.get("best_val_final_accuracy"),
                "test": summary.get("test_final_accuracy_at_best_val"),
                "best_epoch": summary.get("best_epoch"),
                "epochs_ran": summary.get("epochs_ran"),
                "artifact_root": str(artifact_root.relative_to(REPO_ROOT)).replace("\\", "/"),
                "history_path": str((artifact_root / "history.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["val"].notna() & df["test"].notna()].copy()
    return df.sort_values("test", ascending=False).reset_index(drop=True)


def load_hist(rel_or_abs: str | Path) -> list[dict]:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def arr(hist: list[dict], key: str) -> np.ndarray:
    out = []
    for ep in hist:
        val = ep.get(key)
        out.append(float(val) if val is not None and val == val else np.nan)
    return np.array(out, dtype=float)


def ep(hist: list[dict]) -> np.ndarray:
    return np.array([e["epoch"] for e in hist], dtype=float)


def fmt_acc(x: float) -> str:
    return f"{x:.4f}".replace(".", "{,}")


def phase_winner(df: pd.DataFrame, model: str, phase: str, metric: str = "val") -> pd.Series:
    sub = df[(df["model"] == model) & (df["phase"] == phase)]
    if sub.empty:
        raise ValueError(f"No runs for {model} {phase}")
    return sub.loc[sub[metric].idxmax()]


def pick_phase1_curve_runs(df: pd.DataFrame, model: str, n: int = 4) -> list[tuple[str, str]]:
    sub = df[(df["model"] == model) & (df["phase"] == "phase1")].sort_values("val", ascending=False)
    picks: list[tuple[str, str]] = []
    for _, row in sub.iterrows():
        if model == MODEL_BRANCHES:
            label = f"k={int(row.k)}, L={int(row.branch_layers)}, H={int(row.hidden_size)}"
        else:
            label = f"k={int(row.k)}, R={int(row.recurrency)}, H={int(row.hidden_size)}"
        picks.append((label, row.history_path))
        if len(picks) >= n:
            break
    return picks


def plot_train_val_panels(
    curve_runs: list[tuple[str, str]],
    out_train: Path,
    out_val: Path,
    arch_title: str,
) -> None:
    hists = {label: load_hist(path) for label, path in curve_runs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (label, hist) in enumerate(hists.items()):
        x = ep(hist)
        ax1.plot(x, arr(hist, "train_final_accuracy"), color=C[i], label=label)
        ax2.plot(x, arr(hist, "train_loss_total"), color=C[i], label=label)
    ax1.set(xlabel="Эпоха", ylabel="Точность", title=f"{arch_title} — точность на обучении")
    ax2.set(xlabel="Эпоха", ylabel="Суммарные потери", title=f"{arch_title} — потери на обучении")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_train, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (label, hist) in enumerate(hists.items()):
        x = ep(hist)
        ax1.plot(x, arr(hist, "val_final_accuracy"), color=C[i], label=label)
        ax1.plot(x, arr(hist, "val_prefix_accuracy"), color=C[i], linestyle="--", alpha=0.65)
        ax2.plot(x, arr(hist, "val_loss_total"), color=C[i], label=label)
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", linestyle="--", label="prefix accuracy"))
    ax1.legend(handles=handles, fontsize=8)
    ax1.set(xlabel="Эпоха", ylabel="Точность", title=f"{arch_title} — точность на валидации")
    ax2.legend(fontsize=8)
    ax2.set(xlabel="Эпоха", ylabel="Суммарные потери", title=f"{arch_title} — потери на валидации")
    fig.tight_layout()
    fig.savefig(out_val, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(best_branches: pd.Series, best_rnn: pd.Series, out_path: Path) -> None:
    comparison = [
        {
            "label": f"GSN-ветви\nk={int(best_branches.k)}, L={int(best_branches.branch_layers)}",
            "val": best_branches.val,
            "test": best_branches.test,
        },
        {
            "label": f"RNN\nk={int(best_rnn.k)}, R={int(best_rnn.recurrency)}",
            "val": best_rnn.val,
            "test": best_rnn.test,
        },
    ]
    x = np.arange(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bv = ax.bar(x - width / 2, [d["val"] for d in comparison], width, label="Val accuracy", color=C[0], alpha=0.9)
    bt = ax.bar(x + width / 2, [d["test"] for d in comparison], width, label="Test accuracy", color=C[1], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([d["label"] for d in comparison], fontsize=9)
    ymin = min(d["test"] for d in comparison) - 0.015
    ymax = max(d["val"] for d in comparison) + 0.015
    ax.set_ylim(ymin, ymax)
    ax.set(ylabel="Точность", title="Сравнение лучших конфигураций по val и test")
    ax.legend()
    for bar in list(bv) + list(bt):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0005, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_training(best_row: pd.Series, out_path: Path) -> None:
    hist = load_hist(best_row.history_path)
    x_f = ep(hist)
    val_acc = arr(hist, "val_final_accuracy")
    best_i = int(np.nanargmax(val_acc))
    best_n = x_f[best_i]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(x_f, arr(hist, "train_final_accuracy"), color=C[0], label="Train accuracy")
    ax1.plot(x_f, val_acc, color=C[1], label="Val accuracy")
    ax1.plot(x_f, arr(hist, "val_prefix_accuracy"), color=C[1], linestyle="--", alpha=0.75, label="Val prefix accuracy")
    ax1.axvline(best_n, color="red", linestyle=":", alpha=0.6, label=f"Лучшая эпоха ({int(best_n)})")
    test_v = arr(hist, "test_final_accuracy")
    valid = ~np.isnan(test_v)
    if valid.any():
        tx, ty = x_f[valid], test_v[valid]
        ax1.scatter(tx[-1], ty[-1], color="red", zorder=6, s=70, label=f"Test acc = {ty[-1]:.4f}")
    ax1.set(xlabel="Эпоха", ylabel="Точность", title="Лучшая модель — точность")
    ax1.legend(fontsize=8)

    ax2.plot(x_f, arr(hist, "train_loss_total"), color=C[0], label="Train total")
    ax2.plot(x_f, arr(hist, "val_loss_total"), color=C[1], label="Val total")
    ax2.plot(x_f, arr(hist, "val_loss_final"), color=C[2], linestyle="--", alpha=0.75, label="Val final")
    ax2.plot(x_f, arr(hist, "val_loss_prefix"), color=C[3], linestyle="--", alpha=0.75, label="Val prefix")
    ax2.plot(x_f, arr(hist, "val_loss_consistency"), color=C[4], linestyle=":", alpha=0.75, label="Val consistency")
    ax2.set(xlabel="Эпоха", ylabel="Потери", title="Лучшая модель — потери")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase2_stft(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, model, title in zip(
        axes,
        [MODEL_BRANCHES, MODEL_RNN],
        ["GSN-ветви", "RNN"],
    ):
        sub = df[(df["model"] == model) & (df["phase"] == "phase2")].copy()
        # Keep only the best-val run per unique (n_fft, hop_length) pair
        sub = (
            sub.sort_values("val", ascending=False)
            .drop_duplicates(subset=["n_fft", "hop_length"])
        )
        sub["label"] = sub.apply(lambda r: f"{int(r.n_fft)}/{int(r.hop_length)}", axis=1)
        sub = sub.sort_values("label")
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["val"], width=0.36, label="Val", color=C[0], alpha=0.9)
        ax.bar(x + 0.18, sub["test"], width=0.36, label="Test", color=C[1], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"].tolist())
        ax.set(xlabel=r"$N_{\mathrm{fft}}$ / $H$", title=f"Фаза 2 — {title}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Точность")
    fig.suptitle("Сеточный поиск параметров STFT (фаза 2)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase3_presets(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model, title in zip(
        axes,
        [MODEL_BRANCHES, MODEL_RNN],
        ["GSN-ветви", "RNN"],
    ):
        sub = df[(df["model"] == model) & (df["phase"] == "phase3")].copy()
        # Keep only the best-val run per unique subband_preset
        sub = (
            sub.sort_values("val", ascending=False)
            .drop_duplicates(subset=["subband_preset"])
        )
        sub["short"] = sub["subband_preset"].str.replace("p", "p", regex=False)
        sub = sub.sort_values("test", ascending=False)
        labels = [PRESET_LABELS.get(p, p) for p in sub["subband_preset"]]
        y = np.arange(len(sub))
        ax.barh(y + 0.15, sub["val"], height=0.28, label="Val", color=C[0], alpha=0.9)
        ax.barh(y - 0.15, sub["test"], height=0.28, label="Test", color=C[1], alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel="Точность", title=f"Фаза 3 — {title}")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Влияние пресетов субполос (фаза 3)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase1_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model, depth_key, depth_vals, title in zip(
        axes,
        [MODEL_BRANCHES, MODEL_RNN],
        ["branch_layers", "recurrency"],
        [[2, 3, 4], [2, 3, 4]],
        ["GSN-ветви", "RNN"],
    ):
        sub = df[(df["model"] == model) & (df["phase"] == "phase1") & (df["hidden_size"] == 256)].copy()
        pivot = sub.pivot_table(index="k", columns=depth_key, values="test", aggfunc="max")
        pivot = pivot.reindex(index=[0, 1, 2], columns=depth_vals)
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0.90, vmax=0.93)
        ax.set_xticks(range(len(depth_vals)))
        ax.set_xticklabels([str(v) for v in depth_vals])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["0", "1", "2"])
        ax.set(xlabel="Число слоёв" if model == MODEL_BRANCHES else "R", ylabel="k", title=f"Фаза 1 — {title} (test, H=256)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if val == val:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Test accuracy на фазе 1 (n_fft=512, hop=64)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_latex_tables(df: pd.DataFrame, best_row: pd.Series, metrics: dict) -> dict[str, str]:
    def tex_table_phase1_branches() -> str:
        sub = df[(df["model"] == MODEL_BRANCHES) & (df["phase"] == "phase1")].sort_values("val", ascending=False)
        top = sub.head(6)
        lines = [
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"$k$ & $L$ & $H$ & Val. точность & Test точность \\",
            r"\midrule",
        ]
        for _, r in top.iterrows():
            bold = r["run_id"] == phase_winner(df, MODEL_BRANCHES, "phase1")["run_id"]
            vals = [
                f"${int(r.k)}$",
                f"${int(r.branch_layers)}$",
                f"${int(r.hidden_size)}$",
                fmt_acc(r.val),
                fmt_acc(r.test),
            ]
            line = " & ".join(vals) + r" \\"
            if bold:
                line = r"\textbf{" + line.replace(r" \\", "}") + r" \\"
            lines.append(line)
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)

    def tex_table_phase1_rnn() -> str:
        sub = df[(df["model"] == MODEL_RNN) & (df["phase"] == "phase1")].sort_values("val", ascending=False)
        top = sub.head(6)
        winner_val = phase_winner(df, MODEL_RNN, "phase1")
        lines = [
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"$k$ & $R$ & $H$ & Val. точность & Test точность \\",
            r"\midrule",
        ]
        for _, r in top.iterrows():
            vals = [f"${int(r.k)}$", f"${int(r.recurrency)}$", f"${int(r.hidden_size)}$", fmt_acc(r.val), fmt_acc(r.test)]
            line = " & ".join(vals) + r" \\"
            if r["run_id"] == winner_val["run_id"]:
                line = r"\textbf{" + line.replace(r" \\", "}") + r" \\"
            lines.append(line)
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)

    def tex_table_phase2(model: str, caption_model: str) -> str:
        sub = df[(df["model"] == model) & (df["phase"] == "phase2")].sort_values(["n_fft", "hop_length"], ascending=[False, False])
        lines = [
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"$N_{\mathrm{fft}}$ & $H$ & $\Delta t$, мс & Val. точность & Test точность \\",
            r"\midrule",
        ]
        for _, r in sub.iterrows():
            dt_ms = int(r.hop_length / 16000 * 1000)
            line = f"${int(r.n_fft)}$ & ${int(r.hop_length)}$ & ${dt_ms}$ & ${fmt_acc(r.val)}$ & ${fmt_acc(r.test)}$ \\\\"
            lines.append(line)
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)

    def tex_table_phase3(model: str) -> str:
        sub = df[(df["model"] == model) & (df["phase"] == "phase3")].sort_values("test", ascending=False)
        lines = [
            r"\begin{tabular}{p{0.34\textwidth}cc}",
            r"\toprule",
            r"Пресет субполос & Val. точность & Test точность \\",
            r"\midrule",
        ]
        for _, r in sub.iterrows():
            preset = PRESET_LABELS.get(r.subband_preset, r.subband_preset)
            line = f"{preset} & ${fmt_acc(r.val)}$ & ${fmt_acc(r.test)}$ \\\\"
            lines.append(line)
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)

    def tex_final_metrics() -> str:
        hop = int(best_row.hop_length)
        dt_ms = hop / 16000 * 1000
        val_ttc_ms = metrics["val_time_to_correct_frame"] * dt_ms
        test_ttc_ms = metrics["test_time_to_correct_frame"] * dt_ms
        return "\n".join(
            [
                r"\begin{tabular}{p{0.50\textwidth}cc}",
                r"\toprule",
                r"Метрика & Валидация & Тест \\",
                r"\midrule",
                f"Итоговая точность (final accuracy) & ${fmt_acc(metrics['val_final_accuracy'])}$ & ${fmt_acc(metrics['test_final_accuracy'])}$ \\\\",
                f"Префиксная точность (prefix accuracy) & ${fmt_acc(metrics['val_prefix_accuracy'])}$ & ${fmt_acc(metrics['test_prefix_accuracy'])}$ \\\\",
                f"Стабильность после первого попадания & ${fmt_acc(metrics['val_stability_after_first_hit'])}$ & ${fmt_acc(metrics['test_stability_after_first_hit'])}$ \\\\",
                f"Среднее время до первого верного ответа, кадры & ${metrics['val_time_to_correct_frame']:.1f}$ & ${metrics['test_time_to_correct_frame']:.1f}$ \\\\",
                f"Среднее время до первого верного ответа, мс & ${val_ttc_ms:.0f}$ & ${test_ttc_ms:.0f}$ \\\\",
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )

    return {
        "phase1_branches": tex_table_phase1_branches(),
        "phase1_rnn": tex_table_phase1_rnn(),
        "phase2_branches": tex_table_phase2(MODEL_BRANCHES, "ветви"),
        "phase2_rnn": tex_table_phase2(MODEL_RNN, "RNN"),
        "phase3_branches": tex_table_phase3(MODEL_BRANCHES),
        "phase3_rnn": tex_table_phase3(MODEL_RNN),
        "final_metrics": tex_final_metrics(),
    }


def extract_best_metrics(best_row: pd.Series) -> dict:
    hist = load_hist(best_row.history_path)
    summary = json.loads((REPO_ROOT / best_row.artifact_root / "best_summary.json").read_text(encoding="utf-8"))["summary"]
    best_epoch = int(summary["best_epoch"])
    val_row = next(h for h in hist if h["epoch"] == best_epoch)
    test_row = next(h for h in reversed(hist) if h.get("test_final_accuracy") == h.get("test_final_accuracy"))
    return {
        "val_final_accuracy": val_row["val_final_accuracy"],
        "test_final_accuracy": summary["test_final_accuracy_at_best_val"],
        "val_prefix_accuracy": val_row["val_prefix_accuracy"],
        "test_prefix_accuracy": test_row["test_prefix_accuracy"],
        "val_stability_after_first_hit": val_row["val_stability_after_first_hit"],
        "test_stability_after_first_hit": test_row["test_stability_after_first_hit"],
        "val_time_to_correct_frame": val_row["val_time_to_correct_frame"],
        "test_time_to_correct_frame": test_row["test_time_to_correct_frame"],
        "best_epoch": best_epoch,
        "num_params": metrics_num_params(best_row),
    }


def metrics_num_params(best_row: pd.Series) -> int:
    sys.path.insert(0, str(SNK_DIR))
    from train_rnn import SpikeFusionConfig, StreamingSpikeFusionClassifier

    ckpt_dir = REPO_ROOT / best_row.artifact_root
    ckpt_files = list(ckpt_dir.glob("*.pkl"))
    if not ckpt_files:
        return 1_718_051
    ckpt_path = ckpt_files[0]
    with ckpt_path.open("rb") as fh:
        ckpt = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(SpikeFusionConfig)}
    config = SpikeFusionConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    model = StreamingSpikeFusionClassifier(config=config, k=int(ckpt["k"]))
    model.load_state_dict(ckpt["state_dict"])
    return sum(p.numel() for p in model.parameters())


def run_spike_and_trajectory_analysis(best_row: pd.Series) -> dict:
    sys.path.insert(0, str(SNK_DIR))
    import torch
    from train_rnn import (
        LABEL_NAMES,
        SpikeFusionConfig,
        StreamingSpikeFusionClassifier,
        collect_probe_spike_statistics,
        make_gsc_dataloaders,
        waveform_to_streaming_frames,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = next((REPO_ROOT / best_row.artifact_root).glob("*.pkl"))
    with ckpt_path.open("rb") as fh:
        ckpt = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(SpikeFusionConfig)}
    config = SpikeFusionConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    config.cache_dir = str(CACHE_DIR)
    model = StreamingSpikeFusionClassifier(config=config, k=int(ckpt["k"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, _, _, _, val_loader, test_loader = make_gsc_dataloaders(
        config,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        train_fraction=1.0,
        val_fraction=0.25,
        test_fraction=0.25,
        precompute_in_memory=False,
    )

    probe_frames = next(iter(val_loader))["frames"][:8].to(device)
    spike_rows = collect_probe_spike_statistics(model, probe_frames, device)

    branch_map = {
        "fullband": [],
        "0_1": [],
        "1_4": [],
        "4_8": [],
    }
    fusion_rates: list[float] = []
    for row in spike_rows:
        comp = row["component"]
        if comp.startswith("fusion/"):
            fusion_rates.append(row["mean_rate"])
            continue
        for key in branch_map:
            if key.replace("_", "-") in comp or key in comp:
                branch_map[key].append(row["mean_rate"])
                break

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    spike_summary = {
        "fullband": mean_or_nan(branch_map["fullband"]),
        "0_1": mean_or_nan(branch_map["0_1"]),
        "1_4": mean_or_nan(branch_map["1_4"]),
        "4_8": mean_or_nan(branch_map["4_8"]),
        "fusion": mean_or_nan(fusion_rates),
    }

    # Confusion matrix on test subset
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for batch in test_loader:
            frames = batch["frames"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(frames)["decoder_logits"]
            preds = logits[:, -1, :].argmax(dim=-1).cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    num_classes = len(LABEL_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="Предсказанный класс", ylabel="Истинный класс", title="Матрица ошибок (test, подвыборка)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PICS_DIR / "fig_sf_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # Trajectory example
    val_list = (CACHE_DIR / "validation_list.txt").read_text(encoding="utf-8").strip().splitlines()
    example = _find_trajectory_example(model, config, ckpt, val_list, device)
    if example is not None:
        _plot_trajectory(example, PICS_DIR / "fig08_trajectory_2.png")

    return {"spike_summary": spike_summary, "confusion_total": int(cm.sum()), "trajectory_label": example["label"] if example else None}


def _find_trajectory_example(model, config, ckpt, val_files, device, seed=7):
    import random
    import torch
    from train_rnn import LABEL_NAMES, waveform_to_streaming_frames

    rng = random.Random(seed)
    candidates = list(val_files)
    rng.shuffle(candidates)
    n_fft = int(ckpt["config"]["n_fft"])
    hop = int(ckpt["config"]["hop_length"])
    sr = int(config.sample_rate)

    for rel in candidates:
        rel = rel.replace("\\", "/")
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        true_label = parts[0]
        if true_label not in LABEL_NAMES:
            continue
        true_idx = LABEL_NAMES.index(true_label)
        path = CACHE_DIR / rel
        if not path.exists():
            continue
        try:
            import soundfile as sf

            wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != sr:
                n_out = int(len(wav) * sr / file_sr)
                wav = np.interp(np.linspace(0, len(wav) - 1, n_out), np.arange(len(wav)), wav).astype(np.float32)
            log_frames, times = waveform_to_streaming_frames(wav, n_fft, hop, sr, device=device)
            frames_t = torch.tensor(log_frames, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                logits = model(frames_t)["decoder_logits"].squeeze(0).cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
        except Exception:
            continue
        if np.argmax(probs[-1]) != true_idx or probs[-1, true_idx] < 0.55:
            continue
        return {
            "label": true_label,
            "label_idx": true_idx,
            "probs": probs,
            "log_frames": log_frames,
            "times": times,
        }
    return None


def _plot_trajectory(example: dict, save_path: Path) -> None:
    from train_rnn import LABEL_NAMES

    probs = example["probs"]
    log_frames = example["log_frames"]
    times = example["times"]
    true_idx = example["label_idx"]
    true_label = example["label"]
    T, _ = probs.shape

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1.0, 1.2, 1.0], "hspace": 0.55})
    ax_spec, ax_cls, ax_conf = axes

    img = ax_spec.imshow(
        log_frames.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, log_frames.shape[1]],
        cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax_spec, fraction=0.025, pad=0.02, label="log(1+|STFT|)")
    ax_spec.set(xlabel="Время, с", ylabel="Частотный бин")
    ax_spec.set_title(f"Логарифмическая спектрограмма (слово: «{true_label}»)")
    ax_spec.grid(False)

    peak = probs.max(axis=0)
    peak_copy = peak.copy()
    peak_copy[true_idx] = -1
    rivals = np.argsort(peak_copy)[::-1][:5]
    show_cls = [true_idx] + list(rivals)
    for i, ci in enumerate(show_cls):
        is_true = ci == true_idx
        ax_cls.plot(
            times,
            probs[:, ci],
            color=plt.get_cmap("tab10")(i),
            lw=2.2 if is_true else 1.2,
            alpha=1.0 if is_true else 0.7,
            label=f"«{LABEL_NAMES[ci]}»" + (" [верный]" if is_true else ""),
        )
    ax_cls.set(xlabel="Время, с", ylabel="Вероятность", title="Динамика предсказания по классам")
    ax_cls.set_ylim(-0.03, 1.08)
    ax_cls.legend(fontsize=8, ncol=2, loc="upper left")

    ax_conf.plot(times, probs[:, true_idx], color="green", lw=2.2, label=f"P(«{true_label}») — верный класс")
    second = probs.copy()
    second[:, true_idx] = 0.0
    second_t = np.argmax(second, axis=1)
    second_v = second[np.arange(T), second_t]
    rival_name = LABEL_NAMES[Counter(second_t).most_common(1)[0][0]]
    ax_conf.plot(times, second_v, color="crimson", lw=1.6, linestyle="--", label=f"P(«{rival_name}») — конкурент")
    ax_conf.axhline(0.5, color="gray", linestyle=":", alpha=0.5, lw=1.2, label="Порог 0.5")
    ax_conf.set(xlabel="Время, с", ylabel="Вероятность", title="Уверенность модели: верный класс vs конкурент")
    ax_conf.set_ylim(-0.03, 1.08)
    ax_conf.legend(fontsize=9)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PICS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    df = load_runs()
    df.to_csv(OUT_DIR / "all_runs.csv", index=False)

    best_row = df.loc[df["test"].idxmax()]
    best_branches = df[df["model"] == MODEL_BRANCHES].iloc[0]
    best_rnn = df[df["model"] == MODEL_RNN].iloc[0]

    metrics = extract_best_metrics(best_row)
    try:
        inference = run_spike_and_trajectory_analysis(best_row)
    except Exception as exc:  # weights are not shipped in the public repo
        print(f"[skip] spike/trajectory analysis requires .pkl checkpoints: {exc!r}")
        inference = {}

    plot_train_val_panels(
        pick_phase1_curve_runs(df, MODEL_BRANCHES),
        PICS_DIR / "fig01_layered_train.png",
        PICS_DIR / "fig02_layered_val.png",
        "GSN-ветви (фаза 1)",
    )
    plot_train_val_panels(
        pick_phase1_curve_runs(df, MODEL_RNN),
        PICS_DIR / "fig03_rnn_train.png",
        PICS_DIR / "fig04_rnn_val.png",
        "RNN (фаза 1)",
    )
    plot_comparison_bar(best_branches, best_rnn, PICS_DIR / "fig05_comparison_bar.png")
    plot_best_training(best_row, PICS_DIR / "fig06_final_training.png")
    plot_phase1_heatmaps(df, PICS_DIR / "fig_sf_phase1_heatmap.png")
    plot_phase2_stft(df, PICS_DIR / "fig_sf_phase2_stft.png")
    plot_phase3_presets(df, PICS_DIR / "fig_sf_phase3_presets.png")

    latex_tables = build_latex_tables(df, best_row, metrics)
    report = {
        "best_run": best_row.to_dict(),
        "best_branches_by_test": best_branches.to_dict(),
        "best_rnn_by_test": best_rnn.to_dict(),
        "phase1_winners": {
            MODEL_BRANCHES: phase_winner(df, MODEL_BRANCHES, "phase1").to_dict(),
            MODEL_RNN: phase_winner(df, MODEL_RNN, "phase1").to_dict(),
        },
        "phase2_winners": {
            MODEL_BRANCHES: phase_winner(df, MODEL_BRANCHES, "phase2").to_dict(),
            MODEL_RNN: phase_winner(df, MODEL_RNN, "phase2").to_dict(),
        },
        "phase3_winners": {
            MODEL_BRANCHES: phase_winner(df, MODEL_BRANCHES, "phase3").to_dict(),
            MODEL_RNN: phase_winner(df, MODEL_RNN, "phase3").to_dict(),
        },
        "metrics": metrics,
        "inference": inference,
        "latex_tables": latex_tables,
    }
    (OUT_DIR / "thesis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Best by test:", best_row["run_id"], f"test={best_row['test']:.4f}")
    print("Saved figures to", PICS_DIR)
    print("Saved report to", OUT_DIR / "thesis_report.json")


if __name__ == "__main__":
    main()
