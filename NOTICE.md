# Attribution

## Gated Spiking Unit (GSU)

`GSUCell`, `GSULayer`, `EfficientSpikingNeuron` and the triangle surrogate
gradient live in `snn_kws/neurons.py` and are adapted from Spiking-FullSubNet:

- Paper: Hao et al., "Toward Ultralow-Power Neuromorphic Speech Enhancement
  With Spiking-FullSubNet", IEEE TNNLS, 2025.
- Code: https://github.com/haoxiangsnr/spiking-fullsubnet
- License: MIT, Copyright (c) 2023 郝翔

Local changes relative to that cell: `F.linear` instead of explicit `mm` for
autocast/`torch.compile` stability, optional `torch.compile` on the cell
forward, and CUDA-graph-friendly state handling.

## Adaptive LIF (AdLIF)

`LIFCell` / `AdLIFCell` follow the discrete-time RLIF / RadLIF in sparch
(Bittar & Garner 2022), BSD-3-Clause, Idiap Research Institute:

- Paper: https://doi.org/10.3389/fnins.2022.865897
- Code: https://github.com/idiap/sparch (`sparch/models/snns.py`)

Membrane voltage `u` is state. Spike threshold `ϑ` is a fixed hyperparameter
(default 1). Per-neuron `α` (and AdLIF `β, a, b`) are trainable and clamped to
the sparch ranges. Adaptation is a current `w` updated *before* `u`, not a
moving threshold. Recurrent `V` has a zero diagonal. Same `W_ih` / `W_hh`
interface as GSU so `--neuron-type lif` or `adlif` isolates the membrane.

## Original work in this repository

The streaming keyword-spotting pipeline is original:

- STFT frontend and subband branch layout for Google Speech Commands
- Sum-threshold branch fusion with a straight-through estimator
- Layered vs recurrent GSU branches for classification (not speech enhancement)
- Prefix / consistency losses and streaming metrics
- Training loop, CUDA graphs, and the three-phase grid search
