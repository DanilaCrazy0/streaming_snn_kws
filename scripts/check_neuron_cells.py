#!/usr/bin/env python3
"""CPU/GPU sanity check for GSU, LIF, and AdLIF cells (no dataset).

Run on the rented 4090 before the long training job:

    python scripts/check_neuron_cells.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "snn_kws"))

from snn_kws.neurons import (  # noqa: E402
    AdLIFCell,
    GSUCell,
    LIFCell,
    build_cell,
    normalize_neuron_type,
    zeros_state,
)
from train_rnn import SpikeFusionConfig, StreamingSpikeFusionClassifier  # noqa: E402


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def check_cells(device: torch.device) -> None:
    batch, hidden, input_size = 4, 32, 32
    x = torch.randn(batch, input_size, device=device)
    for name, cls in (("gsu", GSUCell), ("lif", LIFCell), ("adlif", AdLIFCell)):
        cell = cls(input_size, hidden).to(device)
        state = zeros_state(batch, hidden, device)
        spikes, new_state = cell(x, state)
        assert spikes.shape == (batch, hidden), (name, spikes.shape)
        assert new_state.hx.shape == (batch, hidden)
        assert new_state.cx.shape == (batch, hidden)
        assert new_state.ax.shape == (batch, hidden)
        loss = spikes.float().sum()
        loss.backward()
        print(
            f"[cell] {name:6s} params={_count_params(cell):7d} "
            f"spike_rate={float(spikes.detach().float().mean()):.3f}",
            flush=True,
        )


def check_rnn_models(device: torch.device) -> None:
    n_fft = 512
    n_freq = n_fft // 2 + 1
    frames = torch.randn(2, 12, n_freq, device=device)
    counts = {}
    for neuron in ("gsu", "lif", "adlif"):
        cfg = SpikeFusionConfig(
            n_fft=n_fft,
            hop_length=64,
            hidden_size=64,
            recurrency=3,
            fusion_hidden_size=64,
            fusion_num_layers=1,
            batch_size=2,
            subband_preset="p3_default",
            neuron_type=normalize_neuron_type(neuron),
        )
        model = StreamingSpikeFusionClassifier(config=cfg, k=2).to(device)
        counts[neuron] = _count_params(model)
        out = model(frames)
        logits = out["decoder_logits"]
        assert logits.shape[0] == 2 and logits.shape[-1] == cfg.num_classes
        logits.sum().backward()
        print(
            f"[rnn]  {neuron:6s} params={counts[neuron]:8d} "
            f"logits={tuple(logits.shape)}",
            flush=True,
        )
    print(
        f"[rnn]  param delta vs GSU: LIF {counts['lif'] - counts['gsu']:+d}, "
        f"AdLIF {counts['adlif'] - counts['gsu']:+d} "
        "(expected: LIF/AdLIF have one gate, not two)",
        flush=True,
    )


def check_cuda_graph(device: torch.device) -> None:
    if device.type != "cuda":
        print("[cuda-graph] skipped (not CUDA)", flush=True)
        return
    cfg = SpikeFusionConfig(
        n_fft=512,
        hop_length=64,
        hidden_size=64,
        recurrency=3,
        fusion_hidden_size=64,
        fusion_num_layers=1,
        batch_size=2,
        subband_preset="p3_default",
        neuron_type="lif",
    )
    model = StreamingSpikeFusionClassifier(config=cfg, k=2).to(device)
    model.band_indices = {name: idx.to(device) for name, idx in model.band_indices.items()}
    frames = torch.randn(2, 8, 257, device=device)
    wrapper = torch.nn.Module()
    wrapper.forward = lambda x: model(x)["decoder_logits"]  # type: ignore[method-assign]
    # Use the same wrapper class the trainer uses if importable.
    try:
        from train_rnn import _LogitsWrapper
    except ImportError:
        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner

            def forward(self, x):
                return self.model(x)["decoder_logits"]

    wrapped = _LogitsWrapper(model)
    wrapped.train()
    try:
        graphed = torch.cuda.make_graphed_callables(wrapped, (frames,))
        out = graphed(frames)
        out.sum().backward()
        print(f"[cuda-graph] LIF capture ok, logits={tuple(out.shape)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[cuda-graph] LIF capture failed: {exc!r}", flush=True)
        raise

    cfg.neuron_type = "adlif"
    model = StreamingSpikeFusionClassifier(config=cfg, k=2).to(device)
    model.band_indices = {name: idx.to(device) for name, idx in model.band_indices.items()}
    wrapped = _LogitsWrapper(model)
    wrapped.train()
    graphed = torch.cuda.make_graphed_callables(wrapped, (frames,))
    out = graphed(frames)
    out.sum().backward()
    print(f"[cuda-graph] AdLIF capture ok, logits={tuple(out.shape)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check LIF/AdLIF cells.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-cuda-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(device), "build_cell": build_cell.__name__}, flush=True)
    check_cells(device)
    check_rnn_models(device)
    if not args.skip_cuda_graph:
        check_cuda_graph(device)
    print("[ok] neuron cells are CUDA-graph compatible", flush=True)


if __name__ == "__main__":
    main()
