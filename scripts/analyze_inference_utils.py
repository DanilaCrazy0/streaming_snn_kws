"""Inference-only Spike-Fusion KWS analyses (journal-article memory/spike analyses).

No SNN weight updates. Load grid-search checkpoints, run val subset, write
REPORT.md / report_summary.json / fig_inf_*.png under analysis_inference/.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SNK_DIR = REPO_ROOT / "snn_kws"
GRID_ROOT = REPO_ROOT / "artifacts" / "grid_search"
RESULTS_GRID = REPO_ROOT / "results" / "grid"
OUT_DIR = REPO_ROOT / "artifacts" / "analysis_inference"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "inference_subset256"
CACHE_DIR = REPO_ROOT / "data" / "google_speech_commands"

if str(SNK_DIR) not in sys.path:
    sys.path.insert(0, str(SNK_DIR))

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

# Checkpoint roles from thesis_report.json / ORCH_EXP_SPEC
CHECKPOINTS = {
    "best_rnn": {
        "role": "Best overall / RNN",
        "variant": "rnn",
        "artifact_root": GRID_ROOT
        / "train_rnn"
        / "bonus_phase3_p3_default_nfft512_hop64_k2_R3_H256",
        "test_acc": 0.9268,
    },
    "best_layered": {
        "role": "Best layered",
        "variant": "layered",
        "artifact_root": GRID_ROOT
        / "train_layered"
        / "final_k2_L2_H256_nfft256_hop64_p7",
        "test_acc": 0.9257,
    },
    "phase2_layered": {
        "role": "Phase2 winner layered",
        "variant": "layered",
        "artifact_root": GRID_ROOT
        / "train_layered"
        / "phase2_nfft256_hop64_k2_L2_H256",
        "test_acc": 0.9223,
    },
    "phase1_rnn": {
        "role": "Phase1 RNN",
        "variant": "rnn",
        "artifact_root": GRID_ROOT
        / "train_rnn"
        / "phase1_nfft512_hop64_k2_R2_H256",
        "test_acc": 0.9264,
    },
}

FIG_NAMES = [
    "fig_inf_spike_rates_branches.png",
    "fig_inf_membrane_dynamics.png",
    "fig_inf_memory_ablation.png",
    "fig_inf_frozen_input.png",
    "fig_inf_trajectory.png",
]


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_pkl(artifact_root: Path) -> Path:
    matches = sorted(artifact_root.glob("*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No .pkl in {artifact_root}")
    return matches[0]


def _filter_config(config_cls, config_dict: dict) -> dict:
    known = {f.name for f in dataclasses.fields(config_cls)}
    return {k: v for k, v in config_dict.items() if k in known}


def load_model(variant: str, ckpt_path: Path, device: torch.device):
    """Load RNN or layered sumfusion checkpoint (filters unknown config keys)."""
    with ckpt_path.open("rb") as fh:
        checkpoint = pickle.load(fh)

    if variant == "rnn":
        from train_rnn import (
            SpikeFusionConfig,
            StreamingSpikeFusionClassifier,
        )

        cfg = SpikeFusionConfig(**_filter_config(SpikeFusionConfig, checkpoint["config"]))
        if "recurrency" not in checkpoint["config"] and "recurrency" in checkpoint:
            cfg.recurrency = int(checkpoint["recurrency"])
        cfg.cache_dir = str(CACHE_DIR)
        model = StreamingSpikeFusionClassifier(config=cfg, k=int(checkpoint["k"])).to(device)
        project_fn = __import__(
            "train_rnn", fromlist=["project_branch_spikes"]
        ).project_branch_spikes
        compute_metrics = __import__(
            "train_rnn", fromlist=["compute_sequence_metrics"]
        ).compute_sequence_metrics
        make_loaders = __import__(
            "train_rnn", fromlist=["make_gsc_dataloaders"]
        ).make_gsc_dataloaders
        label_names = __import__(
            "train_rnn", fromlist=["LABEL_NAMES"]
        ).LABEL_NAMES
    elif variant == "layered":
        from train_layered import (
            SpikeFusionConfig,
            StreamingSpikeFusionClassifier,
        )

        cfg = SpikeFusionConfig(**_filter_config(SpikeFusionConfig, checkpoint["config"]))
        cfg.cache_dir = str(CACHE_DIR)
        model = StreamingSpikeFusionClassifier(
            config=cfg,
            k=int(checkpoint["k"]),
            p=int(checkpoint.get("p", 1)),
        ).to(device)
        project_fn = __import__(
            "train_layered", fromlist=["project_branch_spikes"]
        ).project_branch_spikes
        compute_metrics = __import__(
            "train_layered", fromlist=["compute_sequence_metrics"]
        ).compute_sequence_metrics
        make_loaders = __import__(
            "train_layered", fromlist=["make_gsc_dataloaders"]
        ).make_gsc_dataloaders
        label_names = __import__(
            "train_layered", fromlist=["LABEL_NAMES"]
        ).LABEL_NAMES
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "checkpoint": checkpoint,
        "config": cfg,
        "variant": variant,
        "project_branch_spikes": project_fn,
        "compute_sequence_metrics": compute_metrics,
        "make_gsc_dataloaders": make_loaders,
        "label_names": list(label_names),
        "ckpt_path": ckpt_path,
    }


def make_val_loader(
    bundle: dict,
    *,
    val_limit: int = 256,
    batch_size: int = 8,
    seed: int = 0,
) -> DataLoader:
    """Build a small val subset. RNN loaders use fractions; layered also supports limits."""
    if not CACHE_DIR.is_dir():
        raise FileNotFoundError(
            f"GSC dataset missing at {CACHE_DIR}. "
            "Expected Google Speech Commands under data/google_speech_commands."
        )
    make_loaders = bundle["make_gsc_dataloaders"]
    cfg = bundle["config"]
    cfg.cache_dir = str(CACHE_DIR)
    # Cover val_limit after stratified subsample (~9981 val examples full).
    val_fraction = min(1.0, max(0.02, (val_limit * 2.0) / 9981.0))
    kwargs = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        train_fraction=0.01,
        val_fraction=val_fraction,
        test_fraction=0.01,
        precompute_in_memory=False,
        seed=seed,
    )
    # Layered API also accepts explicit limits; prefer them when available.
    import inspect

    sig = inspect.signature(make_loaders)
    if "val_limit" in sig.parameters:
        kwargs.pop("train_fraction", None)
        kwargs.pop("val_fraction", None)
        kwargs.pop("test_fraction", None)
        kwargs.update(train_limit=8, val_limit=max(val_limit, 8), test_limit=8)

    _, val_ds, _, _, val_loader, _ = make_loaders(cfg, **kwargs)
    if len(val_ds) == 0:
        raise RuntimeError(f"Val dataset empty after load (cache={CACHE_DIR}).")
    n = min(val_limit, len(val_ds))
    if n < len(val_ds):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(val_ds), size=n, replace=False).tolist()
        subset = Subset(val_ds, indices)
    else:
        subset = val_ds
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=val_loader.collate_fn,
        num_workers=0,
        pin_memory=False,
    )


def _branch_step(model, name: str, x: torch.Tensor, state, *, reset: bool):
    """One branch forward_step; handles RNN vs layered signatures."""
    if reset:
        state = None
    head = model.branch_heads[name]
    if hasattr(model, "p"):
        spikes, new_state = head.forward_step(
            x, state=state, recurrent_steps=model.p, return_dynamics=False
        )
    else:
        spikes, new_state = head.forward_step(x, state=state, return_dynamics=False)
    return spikes, new_state


def _fusion_step(model, fusion_pre, fusion_state, *, reset: bool, project_fn):
    if reset:
        fusion_state = None
    spikes, new_state = model.fusion_head.forward_step(
        fusion_pre,
        state=fusion_state,
        recurrent_steps=1,
        return_dynamics=False,
    )
    return spikes, new_state


@torch.inference_mode()
def forward_custom(
    bundle: dict,
    frames: torch.Tensor,
    *,
    reset_state_each_frame: bool = False,
    zero_after: Optional[int] = None,
    corrupt_prefix_frac: Optional[float] = None,
    corrupt_mode: str = "zero",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Frame loop with optional state reset / frozen input / prefix corruption."""
    model = bundle["model"]
    project_fn = bundle["project_branch_spikes"]
    frames = frames.clone()
    bsz, n_frames, _ = frames.shape

    if corrupt_prefix_frac is not None and corrupt_prefix_frac > 0:
        n_corrupt = max(1, int(n_frames * corrupt_prefix_frac))
        if corrupt_mode == "zero":
            frames[:, :n_corrupt, :] = 0.0
        elif corrupt_mode == "shuffle":
            rng = np.random.default_rng(seed)
            order = rng.permutation(n_corrupt)
            frames[:, :n_corrupt, :] = frames[:, order, :]
        else:
            raise ValueError(corrupt_mode)

    if zero_after is not None:
        frames[:, zero_after:, :] = 0.0

    branch_contexts = model._build_branch_contexts(frames)
    branch_states = {name: None for name in model.branch_names}
    fusion_state = None
    fusion_spike_seq = []

    for t in range(n_frames):
        reset = reset_state_each_frame
        current = []
        for name in model.branch_names:
            spikes, branch_states[name] = _branch_step(
                model,
                name,
                branch_contexts[name][:, t, :],
                branch_states[name],
                reset=reset,
            )
            current.append(spikes)
        fusion_pre = project_fn(current)
        fusion_spikes, fusion_state = _fusion_step(
            model, fusion_pre, fusion_state, reset=reset, project_fn=project_fn
        )
        fusion_spike_seq.append(fusion_spikes)

    fusion_spikes_t = torch.stack(fusion_spike_seq, dim=1)
    decoder_logits, decoder_rates = model.rate_decoder(fusion_spikes_t)
    return {
        "fusion_spike_seq": fusion_spikes_t,
        "decoder_logits": decoder_logits,
        "decoder_rates": decoder_rates,
    }



@torch.inference_mode()
def forward_streaming(model, frames: torch.Tensor, return_dynamics: bool = True) -> dict:
    """Native streaming forward (state carried across frames)."""
    return model(frames, return_dynamics=return_dynamics)


@torch.inference_mode()
def forward_reset_each_frame(bundle_or_model, frames: torch.Tensor) -> dict:
    """E3a: call heads with state=None at every time step (no temporal memory).

    Accepts either a load_model bundle dict or a raw model. When given a model,
    reconstructs a minimal bundle using the matching project_branch_spikes.
    """
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        bundle = bundle_or_model
    else:
        model = bundle_or_model
        if hasattr(model, "p"):
            from train_layered import project_branch_spikes
        else:
            from train_rnn import project_branch_spikes
        bundle = {"model": model, "project_branch_spikes": project_branch_spikes}
    return forward_custom(bundle, frames, reset_state_each_frame=True)


def aggregate_metrics(metric_rows: list[dict[str, float]], n_examples: list[int]) -> dict[str, float]:
    total = float(sum(n_examples))
    if total <= 0:
        return {}
    out: dict[str, float] = {}
    keys = metric_rows[0].keys()
    for key in keys:
        vals = [row[key] * n for row, n in zip(metric_rows, n_examples)]
        # time_to_correct may be nan for some batches
        finite = [(v, n) for v, n in zip(vals, n_examples) if np.isfinite(v)]
        if not finite:
            out[key] = float("nan")
        else:
            out[key] = float(sum(v for v, _ in finite) / sum(n for _, n in finite))
    out["num_examples"] = int(total)
    return out


@torch.inference_mode()
def run_eval_pass(
    bundle: dict,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: str = "streaming",
    return_dynamics: bool = False,
    zero_after_frac: Optional[float] = None,
    corrupt_prefix_frac: Optional[float] = None,
    corrupt_mode: str = "zero",
    collect_arrays: bool = False,
) -> dict[str, Any]:
    """Run one evaluation protocol over the loader."""
    model = bundle["model"]
    compute_metrics = bundle["compute_sequence_metrics"]
    warmup = int(getattr(bundle["config"], "warmup_steps", 0) or 0)

    metric_rows: list[dict[str, float]] = []
    n_examples: list[int] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    spike_rate_acc: dict[str, list[float]] = {}
    fusion_cx_fire: list[np.ndarray] = []
    fusion_cx_silent: list[np.ndarray] = []
    mean_abs_cx_t: list[np.ndarray] = []
    frac_border_t: list[np.ndarray] = []
    pop_rate_t: list[np.ndarray] = []
    true_prob_curves: list[np.ndarray] = []
    pred_traj: list[np.ndarray] = []
    raster_example: Optional[dict] = None

    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device)
        labels = batch["labels"].to(device)
        bsz, n_frames, _ = frames.shape
        zero_after = None if zero_after_frac is None else int(n_frames * zero_after_frac)

        if mode == "streaming" and not (
            return_dynamics or zero_after is not None or corrupt_prefix_frac
        ):
            outputs = model(frames, return_dynamics=False)
        elif mode == "streaming" and return_dynamics and zero_after is None and corrupt_prefix_frac is None:
            outputs = model(frames, return_dynamics=True)
        elif mode == "reset":
            outputs = forward_custom(bundle, frames, reset_state_each_frame=True)
        elif mode == "frozen":
            outputs = forward_custom(bundle, frames, zero_after=zero_after)
        elif mode == "prefix_corrupt":
            outputs = forward_custom(
                bundle,
                frames,
                corrupt_prefix_frac=corrupt_prefix_frac,
                corrupt_mode=corrupt_mode,
                seed=batch_idx,
            )
        else:
            # streaming with optional frame edits via custom forward
            outputs = forward_custom(
                bundle,
                frames,
                zero_after=zero_after,
                corrupt_prefix_frac=corrupt_prefix_frac,
                corrupt_mode=corrupt_mode or "zero",
                seed=batch_idx,
            )
            if return_dynamics:
                # re-run native forward on (possibly edited) frames for dynamics
                edited = frames.clone()
                if zero_after is not None:
                    edited[:, zero_after:, :] = 0
                if corrupt_prefix_frac:
                    n_c = max(1, int(n_frames * corrupt_prefix_frac))
                    edited[:, :n_c, :] = 0
                dyn = model(edited, return_dynamics=True)
                outputs = {**outputs, **{k: v for k, v in dyn.items() if k.startswith("branch_") or k.startswith("fusion_layer")}}

        logits = outputs["decoder_logits"]
        metrics = compute_metrics(logits, labels, warmup)
        metric_rows.append(metrics)
        n_examples.append(bsz)

        if collect_arrays or return_dynamics:
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        # Spike rates from branch/fusion spike sequences
        if "branch_spike_seq" in outputs:
            for name, seq in outputs["branch_spike_seq"].items():
                spike_rate_acc.setdefault(name, []).append(float(seq.float().mean().cpu()))
            spike_rate_acc.setdefault("fusion", []).append(
                float(outputs["fusion_spike_seq"].float().mean().cpu())
            )
        else:
            # custom forward only returns fusion spikes — still record fusion
            spike_rate_acc.setdefault("fusion", []).append(
                float(outputs["fusion_spike_seq"].float().mean().cpu())
            )

        if return_dynamics and "fusion_layer_potentials" in outputs:
            # [B, T, L, H] — use last layer
            cx = outputs["fusion_layer_potentials"][:, :, -1, :].float()
            hx = outputs["fusion_layer_spikes"][:, :, -1, :].float()
            fire = hx > 0.5
            if fire.any():
                fusion_cx_fire.append(cx[fire].detach().cpu().numpy())
            if (~fire).any():
                fusion_cx_silent.append(cx[~fire].detach().cpu().numpy())
            mean_abs_cx_t.append(cx.abs().mean(dim=(0, 2)).detach().cpu().numpy())
            frac_border_t.append((cx.abs() < 1.0).float().mean(dim=(0, 2)).detach().cpu().numpy())
            pop_rate_t.append(hx.mean(dim=(0, 2)).detach().cpu().numpy())

            if raster_example is None:
                raster_example = {
                    "fusion_spikes": hx[0].detach().cpu().numpy(),  # [T, H]
                    "branch_spikes": {
                        n: outputs["branch_spike_seq"][n][0].float().detach().cpu().numpy()
                        for n in outputs.get("branch_spike_seq", {})
                    },
                    "label": int(labels[0].item()),
                    "logits": logits[0].detach().cpu().numpy(),
                }

        probs = torch.softmax(logits, dim=-1)
        for i in range(bsz):
            lab = int(labels[i].item())
            true_prob_curves.append(probs[i, :, lab].detach().cpu().numpy())
            pred_traj.append(probs[i].argmax(dim=-1).detach().cpu().numpy())

    summary = aggregate_metrics(metric_rows, n_examples)
    mean_rates = {k: float(np.mean(v)) for k, v in spike_rate_acc.items()}

    result: dict[str, Any] = {
        "metrics": summary,
        "mean_spike_rates": mean_rates,
        "mode": mode,
    }

    if fusion_cx_fire:
        fire_arr = np.concatenate(fusion_cx_fire)
        silent_arr = np.concatenate(fusion_cx_silent) if fusion_cx_silent else np.array([])
        result["fusion_threshold_stats"] = {
            "mean_c_if_fire": float(fire_arr.mean()) if fire_arr.size else float("nan"),
            "mean_c_if_silent": float(silent_arr.mean()) if silent_arr.size else float("nan"),
            "frac_borderline_abs_c_lt_1": float(
                np.mean(np.abs(np.concatenate([fire_arr, silent_arr])) < 1.0)
            )
            if fire_arr.size or silent_arr.size
            else float("nan"),
            "n_fire": int(fire_arr.size),
            "n_silent": int(silent_arr.size),
        }
        # pad time series to common length
        t_len = max(len(x) for x in mean_abs_cx_t)
        def _pad_mean(arrs):
            stacked = np.full((len(arrs), t_len), np.nan)
            for i, a in enumerate(arrs):
                stacked[i, : len(a)] = a
            return np.nanmean(stacked, axis=0)

        result["membrane_timeseries"] = {
            "mean_abs_cx": _pad_mean(mean_abs_cx_t).tolist(),
            "frac_abs_cx_lt_1": _pad_mean(frac_border_t).tolist(),
            "population_rate": _pad_mean(pop_rate_t).tolist(),
        }
        result["raster_example"] = raster_example

    if true_prob_curves:
        # pad to max T
        t_max = max(len(c) for c in true_prob_curves)
        mat = np.full((len(true_prob_curves), t_max), np.nan)
        for i, c in enumerate(true_prob_curves):
            mat[i, : len(c)] = c
        result["true_class_prob_mean"] = np.nanmean(mat, axis=0).tolist()
        # correct vs incorrect by final prediction
        finals = np.array([int(p[-1]) for p in pred_traj])
        labs = torch.cat(all_labels).numpy() if all_labels else np.array([])
        if labs.size:
            correct_mask = finals == labs
            result["true_class_prob_correct"] = (
                np.nanmean(mat[correct_mask], axis=0).tolist() if correct_mask.any() else []
            )
            result["true_class_prob_incorrect"] = (
                np.nanmean(mat[~correct_mask], axis=0).tolist() if (~correct_mask).any() else []
            )
            result["confusion_final"] = _confusion(finals, labs, n_classes=35)

        # trajectory helpers
        ttc_list = []
        stab_list = []
        for pred, lab in zip(pred_traj, labs if labs.size else []):
            hits = np.where(pred == lab)[0]
            if hits.size == 0:
                stab_list.append(0.0)
                continue
            first = int(hits[0])
            ttc_list.append(first)
            stab_list.append(float(np.mean(pred[first:] == lab)))
        result["trajectory_stats"] = {
            "time_to_first_correct_mean": float(np.mean(ttc_list)) if ttc_list else float("nan"),
            "stability_after_hit_mean": float(np.mean(stab_list)) if stab_list else float("nan"),
            "n_with_hit": len(ttc_list),
        }
        if raster_example is not None and "logits" in raster_example:
            result["example_true_prob"] = torch.softmax(
                torch.tensor(raster_example["logits"]), dim=-1
            )[:, raster_example["label"]].numpy().tolist()

    if all_logits and mode in {"streaming", "frozen", "reset", "prefix_corrupt"}:
        # KL between this pass and would need baseline — store stacked for later
        result["_logits_stack"] = torch.cat(all_logits, dim=0)
        result["_labels_stack"] = torch.cat(all_labels, dim=0) if all_labels else None

    return result


def _confusion(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, y in zip(preds, labels):
        cm[int(y), int(p)] += 1
    return cm.tolist()


def branch_rate_correlation(bundle: dict, loader: DataLoader, device: torch.device) -> dict:
    """Per-example mean rates per branch → correlation matrix."""
    model = bundle["model"]
    names = list(model.branch_names)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device)
            out = model(frames, return_dynamics=False)
            bsz = frames.shape[0]
            for i in range(bsz):
                rows.append([float(out["branch_spike_seq"][n][i].float().mean().cpu()) for n in names])
    arr = np.asarray(rows, dtype=float)
    if arr.shape[0] < 2:
        corr = np.eye(len(names))
    else:
        corr = np.corrcoef(arr.T)
        corr = np.nan_to_num(corr, nan=0.0)
    return {"branch_names": names, "correlation": corr.tolist(), "mean_rates": arr.mean(axis=0).tolist()}


def kl_softmax_trajectories(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Mean KL(softmax(a) || softmax(b)) over batch and time."""
    pa = F.softmax(logits_a.float(), dim=-1).clamp_min(1e-8)
    pb = F.softmax(logits_b.float(), dim=-1).clamp_min(1e-8)
    kl = (pa * (pa.log() - pb.log())).sum(dim=-1)
    return float(kl.mean().item())


# ─── E4: aggregate from best_summary.json (no forward) ───────────────────────

def collect_e4_tables() -> dict[str, Any]:
    rows = []
    summary_root = RESULTS_GRID if RESULTS_GRID.exists() else GRID_ROOT
    for summary_path in summary_root.glob("**/best_summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        config = payload.get("config", {})
        root = summary_path.parent
        model = root.parent.name
        run_id = root.name
        phase = "other"
        if run_id.startswith("final_"):
            phase = "final"
        elif "phase3" in run_id:
            phase = "phase3"
        elif "phase2" in run_id:
            phase = "phase2"
        elif "phase1" in run_id:
            phase = "phase1"
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "phase": phase,
                "n_fft": config.get("n_fft", summary.get("n_fft")),
                "hop_length": config.get("hop_length", summary.get("hop_length")),
                "subband_preset": config.get("subband_preset", "p3_default"),
                "k": config.get("k", summary.get("k")),
                "recurrency": config.get("recurrency", summary.get("recurrency")),
                "branch_layers": config.get("branch_layers"),
                "hidden_size": config.get("hidden_size", 128),
                "val": summary.get("best_val_final_accuracy"),
                "test": summary.get("test_final_accuracy_at_best_val"),
            }
        )
    phase2 = [r for r in rows if r["phase"] == "phase2"]
    phase3 = [r for r in rows if r["phase"] == "phase3"]
    phase2_sorted = sorted(phase2, key=lambda r: -(r["test"] or 0))
    phase3_sorted = sorted(phase3, key=lambda r: -(r["test"] or 0))
    return {
        "n_runs_total": len(rows),
        "phase2_top": phase2_sorted[:12],
        "phase3_top": phase3_sorted[:16],
        "interpretation": {
            "stft": "hop=64 (4 ms) dominates; n_fft=512 preferred for RNN, n_fft=256 competitive for layered.",
            "presets": "p3_default (4 branches) remains strongest for RNN; layered gains slightly from p7 band edges.",
        },
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def save_fig(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    shutil.copy2(path, FIGURES_DIR / name)
    plt.close(fig)
    print(f"Saved {name}")
    return path


def plot_e1_spike_rates(rates_rnn: dict, rates_layered: dict, corr: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    def _bar(ax, rates, title):
        # order: fullband, then others, fusion last
        keys = [k for k in rates if k != "fusion"]
        keys = sorted(keys, key=lambda x: (0 if x == "fullband" else 1, x))
        keys = keys + (["fusion"] if "fusion" in rates else [])
        vals = [rates[k] for k in keys]
        labels = [k.replace("_khz", "") for k in keys]
        bars = ax.bar(range(len(keys)), vals, color=C[: len(keys)])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Средняя частота спайков")
        ax.set_title(title)
        ax.set_ylim(0, max(vals + [0.1]) * 1.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    _bar(axes[0], rates_rnn, "RNN (best) — rate по ветвям")
    _bar(axes[1], rates_layered, "Layered (best) — rate по ветвям")
    fig.suptitle("E1. Спайковая активность по ветвям и fusion", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_inf_spike_rates_branches.png")

    # optional correlation heatmap saved alongside report (not in required list but useful)
    if corr:
        fig2, ax = plt.subplots(figsize=(5.5, 4.5))
        mat = np.asarray(corr["correlation"], dtype=float)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        names = [n.replace("_khz", "") for n in corr["branch_names"]]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Корреляция mean-rate между ветвями (RNN)")
        fig2.colorbar(im, ax=ax, fraction=0.046)
        fig2.tight_layout()
        save_fig(fig2, "fig_inf_branch_rate_correlation.png")


def plot_e2_membrane(mem_rnn: dict, mem_layered: dict, thr_rnn: dict, thr_layered: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, mem, title in [
        (axes[0, 0], mem_rnn, "RNN: mean |cx| fusion"),
        (axes[0, 1], mem_layered, "Layered: mean |cx| fusion"),
    ]:
        y = mem.get("mean_abs_cx", [])
        ax.plot(y, color=C[0], label="mean |cx|")
        ax.plot(mem.get("frac_abs_cx_lt_1", []), color=C[1], label="доля |cx|<1")
        ax.set_xlabel("Кадр")
        ax.set_title(title)
        ax.legend()

    for ax, thr, title in [
        (axes[1, 0], thr_rnn, "RNN: cx | fire vs silent"),
        (axes[1, 1], thr_layered, "Layered: cx | fire vs silent"),
    ]:
        ax.bar(
            ["fire", "silent"],
            [thr.get("mean_c_if_fire", np.nan), thr.get("mean_c_if_silent", np.nan)],
            color=[C[0], C[1]],
        )
        ax.set_ylabel("mean cx")
        ax.set_title(title)
        ax.axhline(0, color="gray", lw=0.8)

    fig.suptitle("E2. Динамика мембранных потенциалов fusion", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_inf_membrane_dynamics.png")


def plot_e3a_memory(ablation: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["streaming", "state reset"]
    for ax, arch, title in [
        (axes[0], "rnn", "RNN best"),
        (axes[1], "layered", "Layered best"),
    ]:
        data = ablation.get(arch, {})
        final = [data.get("streaming", {}).get("final_accuracy", 0), data.get("reset", {}).get("final_accuracy", 0)]
        prefix = [data.get("streaming", {}).get("prefix_accuracy", 0), data.get("reset", {}).get("prefix_accuracy", 0)]
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, final, w, label="final acc", color=C[0])
        ax.bar(x + w / 2, prefix, w, label="prefix acc", color=C[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        for i, (f, p) in enumerate(zip(final, prefix)):
            ax.text(i - w / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=8)
            ax.text(i + w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=8)
    fig.suptitle("E3a. Ablation: перенос состояния vs сброс на каждом кадре", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_inf_memory_ablation.png")


def plot_e3b_frozen(frozen: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: accuracy vs freeze fraction
    ax = axes[0]
    fracs = sorted(frozen.get("by_frac", {}).keys())
    accs = [frozen["by_frac"][f]["final_accuracy"] for f in fracs]
    ax.plot([float(f) for f in fracs], accs, marker="o", color=C[0])
    ax.axhline(frozen.get("baseline_final", 0), color="gray", ls="--", label="baseline streaming")
    ax.set_xlabel("Доля слова до заморозки входа (t₀/T)")
    ax.set_ylabel("Final accuracy")
    ax.set_title("E3b. Точность при нулевом входе после t₀")
    ax.legend()

    ax = axes[1]
    curve = frozen.get("true_prob_after_freeze", [])
    if curve:
        ax.plot(curve, color=C[1])
        t0 = frozen.get("example_t0_frame", None)
        if t0 is not None:
            ax.axvline(t0, color="red", ls=":", label=f"t₀={t0}")
        ax.set_xlabel("Кадр")
        ax.set_ylabel("P(true class)")
        ax.set_title("Инерция решения после заморозки входа")
        ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_inf_frozen_input.png")


def plot_e5_trajectory(traj: dict, confusion: Optional[list] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if traj.get("true_class_prob_correct"):
        ax.plot(traj["true_class_prob_correct"], color=C[0], label="correct (final)")
    if traj.get("true_class_prob_incorrect"):
        ax.plot(traj["true_class_prob_incorrect"], color=C[3], label="incorrect (final)")
    if traj.get("true_class_prob_mean"):
        ax.plot(traj["true_class_prob_mean"], color="gray", ls="--", alpha=0.7, label="all mean")
    ax.set_xlabel("Кадр")
    ax.set_ylabel("P(true class)")
    ax.set_title("E5. Траектория вероятности истинного класса")
    ax.legend()

    ax = axes[1]
    cm = np.asarray(confusion or traj.get("confusion_final") or [], dtype=float)
    if cm.size:
        # show only diagonal-normalized top classes by support
        support = cm.sum(axis=1)
        top = np.argsort(-support)[:12]
        sub = cm[np.ix_(top, top)]
        row_sum = sub.sum(axis=1, keepdims=True).clip(min=1)
        sub_n = sub / row_sum
        im = ax.imshow(sub_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("Confusion (top-12 по support)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "no confusion", ha="center")
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig_inf_trajectory.png")


# ─── Report ──────────────────────────────────────────────────────────────────

def write_report(summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    e3a = summary.get("E3a_memory_ablation", {})
    e3b = summary.get("E3b_frozen_input", {})
    e1 = summary.get("E1_spike_rates", {})
    lines = [
        "# Spike-Fusion inference analysis (ORCH_EXP_SPEC)",
        "",
        "Ограничение: **без переобучения** — только загрузка `.pkl` и инференс.",
        "",
        f"- Device: `{summary.get('device')}`",
        f"- Val subset: **{summary.get('val_limit')}** examples",
        f"- Batch size: {summary.get('batch_size')}",
        "",
        "## Чекпоинты",
        "",
    ]
    for key, meta in summary.get("checkpoints", {}).items():
        lines.append(
            f"- **{key}** ({meta.get('role')}): `{meta.get('ckpt_path')}` "
            f"(reported test={meta.get('reported_test_acc')})"
        )
    lines += ["", "## E1. Спайковая активность", ""]
    for arch, rates in e1.items():
        lines.append(f"### {arch}")
        for k, v in rates.items():
            lines.append(f"- `{k}`: **{v:.4f}**")
        lines.append("")

    lines += [
        "## E2. Мембранные потенциалы",
        "",
        f"- RNN mean(c|fire)={summary.get('E2_membrane', {}).get('rnn', {}).get('mean_c_if_fire')}, "
        f"mean(c|silent)={summary.get('E2_membrane', {}).get('rnn', {}).get('mean_c_if_silent')}",
        f"- Layered mean(c|fire)={summary.get('E2_membrane', {}).get('layered', {}).get('mean_c_if_fire')}, "
        f"mean(c|silent)={summary.get('E2_membrane', {}).get('layered', {}).get('mean_c_if_silent')}",
        "",
        "## E3a. State reset ablation (ключевой)",
        "",
    ]
    for arch in ("rnn", "layered"):
        d = e3a.get(arch)
        if not d:
            lines.append(f"### {arch}")
            lines.append("- _(не запускалось в этом прогоне)_")
            lines.append("")
            continue
        s = d.get("streaming") or {}
        r = d.get("reset") or {}
        sf = s.get("final_accuracy")
        sp = s.get("prefix_accuracy")
        rf = r.get("final_accuracy")
        rp = r.get("prefix_accuracy")
        lines.append(f"### {arch}")
        lines.append(
            f"- Streaming: final={sf if sf is None else f'{sf:.4f}'}, "
            f"prefix={sp if sp is None else f'{sp:.4f}'}, "
            f"ttc={s.get('time_to_correct_frame')}"
        )
        lines.append(
            f"- Reset each frame: final={rf if rf is None else f'{rf:.4f}'}, "
            f"prefix={rp if rp is None else f'{rp:.4f}'}, "
            f"ttc={r.get('time_to_correct_frame')}"
        )
        if sf is not None and rf is not None:
            lines.append(f"- Δacc (streaming−reset) = **{sf - rf:.4f}**")
        lines.append(f"- KL(streaming∥reset) softmax traj = {d.get('kl_streaming_vs_reset')}")
        lines.append("")

    lines += [
        "## E3b. Frozen-input continuation",
        "",
        f"- Baseline final: {e3b.get('baseline_final')}",
        f"- By freeze fraction: {json.dumps(e3b.get('by_frac', {}), ensure_ascii=False)}",
        "",
        "## E3c. Prefix corruption (first 25% zeroed)",
        "",
    ]
    e3c = summary.get("E3c_prefix_corrupt", {})
    lines.append(
        f"- Streaming final={e3c.get('streaming_final')}, corrupted final={e3c.get('corrupt_final')}, "
        f"Δ={e3c.get('delta_final')}"
    )
    lines += ["", "## E4. STFT / presets (из best_summary.json)", ""]
    e4 = summary.get("E4_from_summaries", {})
    lines.append(f"- Всего прогонов: {e4.get('n_runs_total')}")
    lines.append(f"- Интерпретация STFT: {e4.get('interpretation', {}).get('stft')}")
    lines.append(f"- Интерпретация presets: {e4.get('interpretation', {}).get('presets')}")
    if e4.get("phase2_top"):
        lines.append("- Phase2 top-3:")
        for row in e4["phase2_top"][:3]:
            lines.append(
                f"  - {row['run_id']}: test={row['test']:.4f} (n_fft={row['n_fft']}, hop={row['hop_length']})"
            )
    lines += ["", "## E5. Prediction trajectory", ""]
    e5 = summary.get("E5_trajectory", {})
    lines.append(f"- time_to_first_correct_mean: {e5.get('time_to_first_correct_mean')}")
    lines.append(f"- stability_after_hit_mean: {e5.get('stability_after_hit_mean')}")
    lines += [
        "",
        "## Фигуры",
        "",
    ]
    for name in summary.get("figures", []):
        lines.append(f"- `{name}` (+ копия в `figures/`)")
    lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj))


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_all(
    *,
    val_limit: int = 256,
    batch_size: int = 8,
    device: Optional[str] = None,
    run_layered_memory: bool = True,
    freeze_fracs: tuple[float, ...] = (0.4, 0.5, 0.75),
) -> dict:
    device_t = resolve_device(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device={device_t}, val_limit={val_limit}, batch_size={batch_size}")

    # Load models
    bundles = {}
    ckpt_meta = {}
    for key in ("best_rnn", "best_layered"):
        info = CHECKPOINTS[key]
        ckpt = find_pkl(info["artifact_root"])
        print(f"Loading {key}: {ckpt}")
        bundles[key] = load_model(info["variant"], ckpt, device_t)
        ckpt_meta[key] = {
            "role": info["role"],
            "ckpt_path": str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
            "reported_test_acc": info["test_acc"],
            "variant": info["variant"],
            "n_fft": bundles[key]["config"].n_fft,
            "hop_length": bundles[key]["config"].hop_length,
            "subband_preset": getattr(bundles[key]["config"], "subband_preset", None),
        }

    loaders = {
        key: make_val_loader(bundles[key], val_limit=val_limit, batch_size=batch_size)
        for key in bundles
    }
    print({k: len(v.dataset) for k, v in loaders.items()})

    summary: dict[str, Any] = {
        "device": str(device_t),
        "val_limit": val_limit,
        "batch_size": batch_size,
        "checkpoints": ckpt_meta,
    }

    # ── E1 + E5 + E2 dynamics on streaming ──
    print("=== E1/E2/E5 streaming passes ===")
    stream = {}
    for key, arch in (("best_rnn", "rnn"), ("best_layered", "layered")):
        print(f"  streaming+dynamics: {key}")
        stream[arch] = run_eval_pass(
            bundles[key],
            loaders[key],
            device_t,
            mode="streaming",
            return_dynamics=True,
            collect_arrays=True,
        )

    summary["E1_spike_rates"] = {
        "rnn": stream["rnn"]["mean_spike_rates"],
        "layered": stream["layered"]["mean_spike_rates"],
    }
    print("  branch correlations (RNN)...")
    corr = branch_rate_correlation(bundles["best_rnn"], loaders["best_rnn"], device_t)
    summary["E1_branch_correlation_rnn"] = corr

    summary["E2_membrane"] = {
        "rnn": stream["rnn"].get("fusion_threshold_stats", {}),
        "layered": stream["layered"].get("fusion_threshold_stats", {}),
        "rnn_timeseries": stream["rnn"].get("membrane_timeseries", {}),
        "layered_timeseries": stream["layered"].get("membrane_timeseries", {}),
    }
    summary["E5_trajectory"] = {
        **stream["rnn"].get("trajectory_stats", {}),
        "true_class_prob_mean": stream["rnn"].get("true_class_prob_mean"),
        "true_class_prob_correct": stream["rnn"].get("true_class_prob_correct"),
        "true_class_prob_incorrect": stream["rnn"].get("true_class_prob_incorrect"),
        "confusion_final": stream["rnn"].get("confusion_final"),
        "streaming_metrics": stream["rnn"]["metrics"],
    }

    plot_e1_spike_rates(
        summary["E1_spike_rates"]["rnn"],
        summary["E1_spike_rates"]["layered"],
        corr,
    )
    plot_e2_membrane(
        summary["E2_membrane"]["rnn_timeseries"],
        summary["E2_membrane"]["layered_timeseries"],
        summary["E2_membrane"]["rnn"],
        summary["E2_membrane"]["layered"],
    )
    plot_e5_trajectory(summary["E5_trajectory"])

    # ── E3a state reset ──
    print("=== E3a state reset ===")
    e3a: dict[str, Any] = {}
    for key, arch in (("best_rnn", "rnn"),) + ((("best_layered", "layered"),) if run_layered_memory else ()):
        print(f"  reset pass: {key}")
        reset_res = run_eval_pass(
            bundles[key], loaders[key], device_t, mode="reset", collect_arrays=True
        )
        base_metrics = stream[arch]["metrics"]
        e3a[arch] = {
            "streaming": base_metrics,
            "reset": reset_res["metrics"],
            "delta_final_accuracy": base_metrics.get("final_accuracy", 0)
            - reset_res["metrics"].get("final_accuracy", 0),
            "delta_prefix_accuracy": base_metrics.get("prefix_accuracy", 0)
            - reset_res["metrics"].get("prefix_accuracy", 0),
        }
        if "_logits_stack" in stream[arch] and "_logits_stack" in reset_res:
            e3a[arch]["kl_streaming_vs_reset"] = kl_softmax_trajectories(
                stream[arch]["_logits_stack"], reset_res["_logits_stack"]
            )
    summary["E3a_memory_ablation"] = e3a
    plot_e3a_memory(e3a)

    # ── E3b frozen input (RNN) ──
    print("=== E3b frozen input ===")
    frozen_by_frac = {}
    frozen_logits_05 = None
    for frac in freeze_fracs:
        print(f"  freeze after {frac}")
        res = run_eval_pass(
            bundles["best_rnn"],
            loaders["best_rnn"],
            device_t,
            mode="frozen",
            zero_after_frac=frac,
            collect_arrays=True,
        )
        frozen_by_frac[str(frac)] = res["metrics"]
        if abs(float(frac) - 0.4) < 1e-9:
            frozen_logits_05 = res.get("_logits_stack")

    # Example trajectory at 0.5 freeze for figure
    ex_loader = DataLoader(
        loaders["best_rnn"].dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=loaders["best_rnn"].collate_fn,
    )
    ex_batch = next(iter(ex_loader))
    frames = ex_batch["frames"].to(device_t)
    labels = ex_batch["labels"].to(device_t)
    t0 = int(frames.shape[1] * 0.4)
    with torch.inference_mode():
        base_logits = bundles["best_rnn"]["model"](frames, return_dynamics=False)["decoder_logits"]
        frozen_logits = forward_custom(bundles["best_rnn"], frames, zero_after=t0)["decoder_logits"]
    lab = int(labels[0].item())
    true_prob_frozen = torch.softmax(frozen_logits[0], dim=-1)[:, lab].cpu().numpy().tolist()
    true_prob_base = torch.softmax(base_logits[0], dim=-1)[:, lab].cpu().numpy().tolist()

    e3b = {
        "baseline_final": stream["rnn"]["metrics"].get("final_accuracy"),
        "by_frac": {
            k: {
                "final_accuracy": v.get("final_accuracy"),
                "prefix_accuracy": v.get("prefix_accuracy"),
            }
            for k, v in frozen_by_frac.items()
        },
        "true_prob_after_freeze": true_prob_frozen,
        "true_prob_baseline_example": true_prob_base,
        "example_t0_frame": t0,
        "example_label": lab,
    }
    if frozen_logits_05 is not None and "_logits_stack" in stream["rnn"]:
        e3b["kl_streaming_vs_frozen_0.4"] = kl_softmax_trajectories(
            stream["rnn"]["_logits_stack"], frozen_logits_05
        )
    summary["E3b_frozen_input"] = e3b
    plot_e3b_frozen(e3b)

    # ── E3c prefix corruption ──
    print("=== E3c prefix corrupt ===")
    corrupt = run_eval_pass(
        bundles["best_rnn"],
        loaders["best_rnn"],
        device_t,
        mode="prefix_corrupt",
        corrupt_prefix_frac=0.25,
        corrupt_mode="zero",
        collect_arrays=True,
    )
    summary["E3c_prefix_corrupt"] = {
        "streaming_final": stream["rnn"]["metrics"].get("final_accuracy"),
        "corrupt_final": corrupt["metrics"].get("final_accuracy"),
        "delta_final": stream["rnn"]["metrics"].get("final_accuracy", 0)
        - corrupt["metrics"].get("final_accuracy", 0),
        "streaming_prefix": stream["rnn"]["metrics"].get("prefix_accuracy"),
        "corrupt_prefix": corrupt["metrics"].get("prefix_accuracy"),
        "kl_streaming_vs_corrupt": kl_softmax_trajectories(
            stream["rnn"]["_logits_stack"], corrupt["_logits_stack"]
        )
        if "_logits_stack" in stream["rnn"] and "_logits_stack" in corrupt
        else None,
    }

    # ── E4 ──
    print("=== E4 from summaries ===")
    summary["E4_from_summaries"] = collect_e4_tables()

    # Strip heavy tensors before JSON
    for arch in list(summary.get("E3a_memory_ablation", {})):
        pass
    # clean nested _logits from stream copies — already not in summary except via metrics

    figures = [n for n in FIG_NAMES if (OUT_DIR / n).exists()]
    # also correlation if present
    extra = "fig_inf_branch_rate_correlation.png"
    if (OUT_DIR / extra).exists():
        figures.append(extra)
    summary["figures"] = figures

    # Drop any accidental tensors
    def _strip(o):
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, list):
            return [_strip(v) for v in o]
        if isinstance(o, torch.Tensor):
            return None
        return o

    summary = _strip(summary)
    write_report(summary)
    print("Done.")
    return summary


if __name__ == "__main__":
    run_all(val_limit=256, batch_size=8)
