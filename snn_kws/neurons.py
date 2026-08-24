"""Spiking cells used by the streaming KWS models.

CUDA-graph constraints (must stay true for every cell):
- no ``.item()`` / host syncs in forward or surrogate backward
- ``F.linear`` instead of ``mm`` + bias (bf16 autocast + compile)
- state tensors are returned, never mutated in-place
- no data-dependent Python control flow

GSU is the original Hao et al. cell (Spiking-FullSubNet). LIF and AdLIF follow
Bittar & Garner 2022 (sparch / ED-sKWS baseline) so a drop-in swap isolates the
gated membrane from a standard leaky / adaptive leaky neuron.
"""

from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


NEURON_TYPES = ("gsu", "lif", "adlif")

# Bittar & Garner 2022, Table 1 / Eqs. (7)–(9): physiologically plausible
# discrete-time ranges at dt = 1 (one STFT frame / one micro-step).
LIF_ALPHA_RANGE = (0.60, 0.96)
ADLIF_ALPHA_RANGE = (0.60, 0.96)
ADLIF_BETA_RANGE = (0.96, 0.99)
ADLIF_A_RANGE = (-1.0, 1.0)
ADLIF_B_RANGE = (0.0, 2.0)
SPIKE_THRESHOLD = 1.0

MemoryState = namedtuple("MemoryState", ["hx", "cx", "ax"])

_COMPILE_CELLS = False


def set_compile_cells(enabled: bool) -> None:
    global _COMPILE_CELLS
    _COMPILE_CELLS = bool(enabled)


def compile_cells_enabled() -> bool:
    return bool(_COMPILE_CELLS)


def normalize_neuron_type(neuron_type: str) -> str:
    key = str(neuron_type).strip().lower()
    aliases = {
        "gsu": "gsu",
        "gsn": "gsu",
        "gating": "gsu",
        "lif": "lif",
        "alif": "adlif",
        "adlif": "adlif",
        "ad_lif": "adlif",
        "adaptive_lif": "adlif",
    }
    if key not in aliases:
        raise ValueError(
            f"unknown neuron_type {neuron_type!r}; expected one of {NEURON_TYPES}"
        )
    return aliases[key]


def zeros_state(batch_size: int, hidden_size: int, device: torch.device) -> MemoryState:
    return MemoryState(
        torch.zeros(batch_size, hidden_size, device=device),
        torch.zeros(batch_size, hidden_size, device=device),
        torch.zeros(batch_size, hidden_size, device=device),
    )


def _range_sigmoid(unconstrained: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return float(low) + (float(high) - float(low)) * torch.sigmoid(unconstrained)


class TriangleSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        # Store gamma as a plain Python float on ctx instead of a saved tensor.
        # The previous version read it back via params[0].item() in backward,
        # which forces a GPU->CPU sync. That sync breaks CUDA graph capture
        # (and causes a torch.compile graph break). Math is unchanged.
        ctx.save_for_backward(input)
        ctx.gamma = float(gamma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate = (1.0 / (gamma * gamma)) * (gamma - inp.abs()).clamp(min=0)
        return grad_output * surrogate, None


triangle_spike = TriangleSurrogate.apply


class GSUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        shared_weights: bool = False,
        bn: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -stdv, stdv)

    def expanded_weights(self):
        if self.shared_weights:
            return self.weight_ih.repeat(2, 1), self.weight_hh.repeat(2, 1)
        return self.weight_ih, self.weight_hh

    def forward(self, input: torch.Tensor, state: MemoryState):
        hx, cx = state.hx, state.cx
        weight_ih, weight_hh = self.expanded_weights()
        # Use F.linear instead of manual mm + bias so that autocast casts the
        # bias consistently with the matmul inputs. The manual form lets
        # torch.compile fuse mm + fp32-bias into a single addmm with mismatched
        # dtypes under bf16 autocast, which raises a dtype error. Math is
        # identical: F.linear(x, W, b) == x @ W.t() + b.
        gates = F.linear(input, weight_ih, self.bias_ih) + F.linear(hx, weight_hh)
        forget_gate, cell_gate = gates.chunk(2, dim=1)
        lam = torch.sigmoid(forget_gate)
        cy = lam * cx + (1.0 - lam) * cell_gate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = triangle_spike(cy)
        ax = state.ax if state.ax is not None else torch.zeros_like(cy)
        return hy, MemoryState(hy, cy, ax)


class LIFCell(nn.Module):
    """Current-based LIF with per-neuron leak and subtractive reset.

    Same recurrent interface as GSU (``W_ih``, ``W_hh``), but the membrane is
    a leaky integrator instead of a learned forget gate:

        I = W_ih x + W_hh h + b
        u_pre = α ⊙ u + (1-α) ⊙ I
        z = H(u_pre)          # threshold 0, same surrogate as GSU
        u = u_pre − z         # subtractive reset of 1

    ``α`` is sigmoid-mapped into ``LIF_ALPHA_RANGE``. ``shared_weights`` / ``bn``
    are accepted for constructor compatibility and ignored (bn is never used
    in this codebase).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        shared_weights: bool = False,
        bn: bool = False,
    ):
        super().__init__()
        del shared_weights, bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_strength = float(SPIKE_THRESHOLD)
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(hidden_size))
        self.leak_param = Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.uniform_(self.bias_ih, -stdv, stdv)
        nn.init.zeros_(self.leak_param)

    def _alpha(self) -> torch.Tensor:
        return _range_sigmoid(self.leak_param, *LIF_ALPHA_RANGE)

    def forward(self, input: torch.Tensor, state: MemoryState):
        hx, cx = state.hx, state.cx
        current = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh)
        alpha = self._alpha()
        u_pre = alpha * cx + (1.0 - alpha) * current
        # Threshold 0 matches GSU's triangle_spike(cy) so random init is not silent.
        hy = triangle_spike(u_pre)
        cy = u_pre - hy * self.reset_strength
        ax = state.ax if state.ax is not None else torch.zeros_like(cy)
        return hy, MemoryState(hy, cy, ax)


class AdLIFCell(nn.Module):
    """Adaptive LIF (Bittar & Garner 2022) with subthreshold + spike-triggered adaptation.

        I = W_ih x + W_hh h + b
        u_pre = α ⊙ u + (1-α) ⊙ I − w
        z = H(u_pre − θ)
        u = u_pre − z θ
        w = β ⊙ w + a ⊙ u + b_adapt ⊙ z_prev

    ``α, β, a, b_adapt`` are per-neuron and mapped into the sparch ranges.
    Adaptation ``w`` lives in ``MemoryState.ax``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        shared_weights: bool = False,
        bn: bool = False,
    ):
        super().__init__()
        del shared_weights, bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_strength = float(SPIKE_THRESHOLD)
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(hidden_size))
        self.alpha_param = Parameter(torch.zeros(hidden_size))
        self.beta_param = Parameter(torch.zeros(hidden_size))
        self.a_param = Parameter(torch.zeros(hidden_size))
        # sigmoid(-2) * 2 ≈ 0.24, a mild spike-triggered jump at init
        self.b_param = Parameter(torch.full((hidden_size,), -2.0))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.uniform_(self.bias_ih, -stdv, stdv)
        nn.init.zeros_(self.alpha_param)
        nn.init.zeros_(self.beta_param)
        nn.init.zeros_(self.a_param)
        nn.init.constant_(self.b_param, -2.0)

    def _alpha(self) -> torch.Tensor:
        return _range_sigmoid(self.alpha_param, *ADLIF_ALPHA_RANGE)

    def _beta(self) -> torch.Tensor:
        return _range_sigmoid(self.beta_param, *ADLIF_BETA_RANGE)

    def _a(self) -> torch.Tensor:
        low, high = ADLIF_A_RANGE
        return _range_sigmoid(self.a_param, low, high)

    def _b(self) -> torch.Tensor:
        return _range_sigmoid(self.b_param, *ADLIF_B_RANGE)

    def forward(self, input: torch.Tensor, state: MemoryState):
        hx, cx = state.hx, state.cx
        wx = state.ax if state.ax is not None else torch.zeros_like(cx)
        current = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh)
        alpha = self._alpha()
        u_pre = alpha * cx + (1.0 - alpha) * current - wx
        hy = triangle_spike(u_pre)
        cy = u_pre - hy * self.reset_strength
        wy = self._beta() * wx + self._a() * cx + self._b() * hx
        return hy, MemoryState(hy, cy, wy)


CELL_REGISTRY: dict[str, type[nn.Module]] = {
    "gsu": GSUCell,
    "lif": LIFCell,
    "adlif": AdLIFCell,
}


def get_cell_class(neuron_type: str) -> type[nn.Module]:
    return CELL_REGISTRY[normalize_neuron_type(neuron_type)]


def build_cell(
    neuron_type: str,
    input_size: int,
    hidden_size: int,
    shared_weights: bool = False,
    bn: bool = False,
) -> nn.Module:
    cell_cls = get_cell_class(neuron_type)
    return cell_cls(input_size, hidden_size, shared_weights, bn)


class GSULayer(nn.Module):
    def __init__(self, cell_cls, *cell_args):
        super().__init__()
        self.cell = cell_cls(*cell_args)
        if _COMPILE_CELLS:
            # Compile the bound forward method rather than wrapping the module,
            # so the cell's parameters and state_dict keys stay unchanged and
            # checkpoints remain compatible with a non-compiled reload.
            self.cell.forward = torch.compile(self.cell.forward)

    def forward(self, input_seq: torch.Tensor, state: MemoryState):
        outputs = []
        current_state = state
        for time_idx in range(input_seq.size(0)):
            out, current_state = self.cell(input_seq[time_idx], current_state)
            outputs.append(out)
        return torch.stack(outputs, dim=0), current_state


class EfficientSpikingNeuron(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        shared_weights: bool = False,
        bn: bool = False,
        neuron_type: str = "gsu",
    ):
        super().__init__()
        cell_cls = get_cell_class(neuron_type)
        layers = [GSULayer(cell_cls, input_size, hidden_size, shared_weights, bn)]
        for _ in range(num_layers - 1):
            layers.append(GSULayer(cell_cls, hidden_size, hidden_size, shared_weights, bn))
        self.layers = nn.ModuleList(layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neuron_type = normalize_neuron_type(neuron_type)

    def forward(self, input_seq: torch.Tensor, states: list[MemoryState]):
        output = input_seq
        new_states = []
        all_layer_outputs = [input_seq]
        for layer, state in zip(self.layers, states):
            output, new_state = layer(output, state)
            new_states.append(new_state)
            all_layer_outputs.append(output)
        return output, new_states, all_layer_outputs
