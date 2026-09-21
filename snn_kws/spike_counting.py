"""Per-example spike counting for the streaming KWS models.

Used for the new Table 2 "spikes" column: total number of spikes emitted by
all spiking layers of the network while processing one utterance.

What is counted
---------------
- Every branch layer: ``branch_layer_spikes[name]`` with shape [B, T, D, H],
  where D is the layer depth (L stacked layers for the layered/ff model, R
  micro-steps for the recurrent model). All D positions are counted, so a
  recurrent model with R>1 reports proportionally more spikes at the same
  parameter count — this is exactly the effect the table should show.
- Every fusion layer: ``fusion_layer_spikes`` with shape [B, T, L_f, H].

The binarized fusion *input* (majority-threshold projection) is not a neuron
output and is not counted.

Statistics are accumulated as running sums / sums of squares over the loader
(validation set per the experiment plan), so no per-example tensors are stored.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch


@torch.inference_mode()
def compute_spike_count_statistics(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> dict[str, float]:
    """Count spikes per example over ``loader`` for a trained ``model``.

    Returns a dict with mean/std (population over examples) of total spikes
    per example, the branch/fusion breakdown, and per-frame normalization.
    """
    was_training = model.training
    model.eval()

    use_autocast = autocast_dtype is not None and device.type == "cuda"
    non_blocking = device.type == "cuda"

    n_examples = 0
    total_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_sumsq = torch.zeros((), dtype=torch.float64, device=device)
    branch_sum = torch.zeros((), dtype=torch.float64, device=device)
    fusion_sum = torch.zeros((), dtype=torch.float64, device=device)
    frames_per_example = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        frames = batch["frames"].to(device, non_blocking=non_blocking)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = model(frames, return_dynamics=True)

        batch_branch = torch.zeros(frames.shape[0], dtype=torch.float64, device=device)
        for tensor in outputs["branch_layer_spikes"].values():
            # [B, T, D, H] -> per-example spike count [B]
            batch_branch += tensor.double().sum(dim=(1, 2, 3))
        batch_fusion = outputs["fusion_layer_spikes"].double().sum(dim=(1, 2, 3))
        batch_total = batch_branch + batch_fusion

        total_sum += batch_total.sum()
        total_sumsq += (batch_total * batch_total).sum()
        branch_sum += batch_branch.sum()
        fusion_sum += batch_fusion.sum()
        n_examples += frames.shape[0]
        frames_per_example = int(frames.shape[1])

        del outputs, batch_branch, batch_fusion, batch_total

    if was_training:
        model.train()

    if n_examples == 0:
        return {
            "num_examples": 0,
            "frames_per_example": 0,
            "spikes_per_example_mean": float("nan"),
            "spikes_per_example_std": float("nan"),
            "branch_spikes_per_example_mean": float("nan"),
            "fusion_spikes_per_example_mean": float("nan"),
            "spikes_per_frame_mean": float("nan"),
        }

    mean = float(total_sum / n_examples)
    # Population std over examples (ddof=0): describes the spread of the
    # per-example spike count itself, not an estimator of a larger population.
    var = float(total_sumsq / n_examples) - mean * mean
    std = max(var, 0.0) ** 0.5
    return {
        "num_examples": int(n_examples),
        "frames_per_example": int(frames_per_example),
        "spikes_per_example_mean": mean,
        "spikes_per_example_std": std,
        "branch_spikes_per_example_mean": float(branch_sum / n_examples),
        "fusion_spikes_per_example_mean": float(fusion_sum / n_examples),
        "spikes_per_frame_mean": mean / max(frames_per_example, 1),
    }
