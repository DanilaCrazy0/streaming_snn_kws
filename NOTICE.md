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

`LIFCell` / `AdLIFCell` follow the discrete-time LIF / adLIF used by Bittar &
Garner 2022 (sparch / ED-sKWS baseline): per-neuron leak in \(α ∈ [0.60, 0.96]\),
AdLIF also learns adaptation \(β, a, b\). Same recurrent \(W_{ih}, W_{hh}\)
interface as GSU so a training run with `--neuron-type lif` or `adlif` isolates
the gated membrane from a standard leaky / adaptive neuron.

## Original work in this repository

The streaming keyword-spotting pipeline is original:

- STFT frontend and subband branch layout for Google Speech Commands
- Sum-threshold branch fusion with a straight-through estimator
- Layered vs recurrent GSU branches for classification (not speech enhancement)
- Prefix / consistency losses and streaming metrics
- Training loop, CUDA graphs, and the three-phase grid search
