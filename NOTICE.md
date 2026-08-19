# Attribution

## Gated Spiking Unit (GSU)

`GSUCell`, `GSULayer`, `EfficientSpikingNeuron` and the triangle surrogate
gradient are adapted from Spiking-FullSubNet:

- Paper: Hao et al., "Toward Ultralow-Power Neuromorphic Speech Enhancement
  With Spiking-FullSubNet", IEEE TNNLS, 2025.
- Code: https://github.com/haoxiangsnr/spiking-fullsubnet
- License: MIT, Copyright (c) 2023 郝翔

Local changes relative to that cell: `F.linear` instead of explicit `mm` for
autocast/`torch.compile` stability, optional `torch.compile` on the cell
forward, and CUDA-graph-friendly state handling.

## Original work in this repository

The streaming keyword-spotting pipeline is original:

- STFT frontend and subband branch layout for Google Speech Commands
- Sum-threshold branch fusion with a straight-through estimator
- Layered vs recurrent GSU branches for classification (not speech enhancement)
- Prefix / consistency losses and streaming metrics
- Training loop, CUDA graphs, and the three-phase grid search
