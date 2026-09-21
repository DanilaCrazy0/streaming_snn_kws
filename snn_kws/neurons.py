"""Spiking cells used by the streaming KWS models.

CUDA-graph constraints (must stay true for every cell):
- no ``.item()`` / host syncs in forward or surrogate backward
- ``F.linear`` instead of ``mm`` + bias (bf16 autocast + compile)
- state tensors are returned, never mutated in-place
- no data-dependent Python control flow

GSU is the original Hao et al. cell (Spiking-FullSubNet). LIF and AdLIF follow
the discrete-time (R)LIF / (R)adLIF of Bittar & Garner 2022 as coded in sparch
(https://github.com/idiap/sparch ``sparch/models/snns.py``), so swapping the
cell isolates the gated membrane from a standard leaky / adaptive leaky neuron.

State vs hyperparameters
------------------------
``MemoryState.cx`` is the membrane potential *state* ``u`` (not a hyperparameter).
The LIF hyperparameter on that voltage is the spike threshold ``ϑ``
(``--spike-threshold``, default 1). The membrane *time constant* enters as
``α = exp(−Δt / τ_u)``, learned per neuron inside a physiological range.
AdLIF adds a slower adaptation current ``w`` (``MemoryState.ax``) with
``β, a, b``.
"""

from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


NEURON_TYPES = ("gsu", "lif", "adlif")

# sparch LIFLayer / adLIFLayer: α = exp(−Δt/τ_u) with τ_u ∈ [5, 25] ms, Δt = 1 ms.
# (The paper text also quotes the slightly wider interval [0.60, 0.96]; the
# released trainer that produced the speech-command numbers uses these bounds.)
LIF_ALPHA_RANGE = (math.exp(-1.0 / 5.0), math.exp(-1.0 / 25.0))
ADLIF_ALPHA_RANGE = LIF_ALPHA_RANGE
# β = exp(−Δt/τ_w) with τ_w ∈ [30, 120] ms.
ADLIF_BETA_RANGE = (math.exp(-1.0 / 30.0), math.exp(-1.0 / 120.0))
ADLIF_A_RANGE = (-1.0, 1.0)
ADLIF_B_RANGE = (0.0, 2.0)
DEFAULT_SPIKE_THRESHOLD = 1.0
# Safety nets only; they must not bite in the typical regime. GSU stays finite
# via its sigmoid forget gate; LIF/AdLIF can still overflow under this recipe
# because there is no batch-norm on the current (sparch's default).
STATE_CLAMP = 20.0
CURRENT_CLAMP = 50.0


def _clamp(tensor: torch.Tensor, limit: float) -> torch.Tensor:
    return tensor.clamp(-float(limit), float(limit))


def _clamp_range(tensor: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    low, high = bounds
    return tensor.clamp(float(low), float(high))


def _zero_diag(weight: torch.Tensor) -> torch.Tensor:
    """Drop self-excitation on ``V`` (Bittar Eq. 12 / sparch RLIF, RadLIF)."""
    return weight - torch.diag(torch.diagonal(weight))


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
        "rlif": "lif",
        "alif": "adlif",
        "adlif": "adlif",
        "ad_lif": "adlif",
        "radlif": "adlif",
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
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    ):
        super().__init__()
        del spike_threshold  # GSU spikes at 0 on the gated membrane, not at ϑ.
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
        gates = F.linear(input, weight_ih, self.bias_ih) + F.linear(hx, weight_hh)
        forget_gate, cell_gate = gates.chunk(2, dim=1)
        lam = torch.sigmoid(forget_gate)
        cy = lam * cx + (1.0 - lam) * cell_gate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = triangle_spike(cy)
        ax = state.ax if state.ax is not None else torch.zeros_like(cy)
        return hy, MemoryState(hy, cy, ax)


class _RecurrentLIFMixin:
    """Shared affine current for RLIF / RadLIF (zero diagonal on ``V``)."""

    def _input_current(self, input: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        current = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(
            spikes, _zero_diag(self.weight_hh)
        )
        return _clamp(current, CURRENT_CLAMP)


class LIFCell(_RecurrentLIFMixin, nn.Module):
    """Recurrent LIF (sparch ``RLIFLayer``).

    ``u`` is state. ``ϑ`` is a fixed hyperparameter. ``α`` (membrane leak /
    time constant) is learned per neuron and clamped to ``LIF_ALPHA_RANGE``.

        I[t] = W_ih x[t] + V s[t-1] + b     # V_ii = 0
        u[t] = α ⊙ (u[t-1] − s[t-1] ϑ) + (1−α) ⊙ I[t]
        s[t] = H(u[t] − ϑ)

    Reset uses the *previous* spike, leaked with the membrane — not a
    same-step subtract-after-integrate, and not GSU's sigmoid interpolation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        shared_weights: bool = False,
        bn: bool = False,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    ):
        super().__init__()
        del shared_weights, bn
        if float(spike_threshold) <= 0.0:
            raise ValueError(f"spike_threshold must be > 0, got {spike_threshold}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.spike_threshold = float(spike_threshold)
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(hidden_size))
        self.alpha = Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.uniform_(self.bias_ih, -stdv, stdv)
        nn.init.uniform_(self.alpha, *LIF_ALPHA_RANGE)

    def _alpha(self) -> torch.Tensor:
        return _clamp_range(self.alpha, LIF_ALPHA_RANGE)

    def forward(self, input: torch.Tensor, state: MemoryState):
        spikes_prev, membrane_prev = state.hx, state.cx
        current = self._input_current(input, spikes_prev)
        alpha = self._alpha()
        threshold = self.spike_threshold
        membrane = alpha * (membrane_prev - spikes_prev * threshold) + (1.0 - alpha) * current
        membrane = _clamp(membrane, STATE_CLAMP)
        spikes = triangle_spike(membrane - threshold)
        ax = state.ax if state.ax is not None else torch.zeros_like(membrane)
        return spikes, MemoryState(spikes, membrane, ax)


class AdLIFCell(_RecurrentLIFMixin, nn.Module):
    """Recurrent adaptive LIF (sparch ``RadLIFLayer`` / Bittar Eqs. 7–9).

    Extra state ``w`` (``MemoryState.ax``) is an adaptation *current*, not a
    moving threshold. It is updated from the *previous* membrane and spike
    *before* the new ``u`` is computed:

        w[t] = β ⊙ w[t-1] + a ⊙ u[t-1] + b ⊙ s[t-1]
        I[t] = W_ih x[t] + V s[t-1] + b_syn     # V_ii = 0
        u[t] = α ⊙ (u[t-1] − s[t-1] ϑ) + (1−α) ⊙ (I[t] − w[t])
        s[t] = H(u[t] − ϑ)

    ``α, β, a, b`` are per-neuron and clamped to the sparch ranges. ``ϑ`` is
    the same fixed hyperparameter as in LIF.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        shared_weights: bool = False,
        bn: bool = False,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    ):
        super().__init__()
        del shared_weights, bn
        if float(spike_threshold) <= 0.0:
            raise ValueError(f"spike_threshold must be > 0, got {spike_threshold}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.spike_threshold = float(spike_threshold)
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(hidden_size))
        self.alpha = Parameter(torch.empty(hidden_size))
        self.beta = Parameter(torch.empty(hidden_size))
        self.a = Parameter(torch.empty(hidden_size))
        self.b = Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        nn.init.uniform_(self.weight_ih, -stdv, stdv)
        nn.init.uniform_(self.weight_hh, -stdv, stdv)
        nn.init.uniform_(self.bias_ih, -stdv, stdv)
        nn.init.uniform_(self.alpha, *ADLIF_ALPHA_RANGE)
        nn.init.uniform_(self.beta, *ADLIF_BETA_RANGE)
        nn.init.uniform_(self.a, *ADLIF_A_RANGE)
        nn.init.uniform_(self.b, *ADLIF_B_RANGE)

    def _alpha(self) -> torch.Tensor:
        return _clamp_range(self.alpha, ADLIF_ALPHA_RANGE)

    def _beta(self) -> torch.Tensor:
        return _clamp_range(self.beta, ADLIF_BETA_RANGE)

    def _a(self) -> torch.Tensor:
        return _clamp_range(self.a, ADLIF_A_RANGE)

    def _b(self) -> torch.Tensor:
        return _clamp_range(self.b, ADLIF_B_RANGE)

    def forward(self, input: torch.Tensor, state: MemoryState):
        spikes_prev, membrane_prev = state.hx, state.cx
        adapt_prev = state.ax if state.ax is not None else torch.zeros_like(membrane_prev)
        current = self._input_current(input, spikes_prev)
        threshold = self.spike_threshold

        adapt = self._beta() * adapt_prev + self._a() * membrane_prev + self._b() * spikes_prev
        adapt = _clamp(adapt, STATE_CLAMP)
        alpha = self._alpha()
        membrane = alpha * (membrane_prev - spikes_prev * threshold) + (1.0 - alpha) * (
            current - adapt
        )
        membrane = _clamp(membrane, STATE_CLAMP)
        spikes = triangle_spike(membrane - threshold)
        return spikes, MemoryState(spikes, membrane, adapt)


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
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
) -> nn.Module:
    cell_cls = get_cell_class(neuron_type)
    return cell_cls(
        input_size,
        hidden_size,
        shared_weights,
        bn,
        spike_threshold=float(spike_threshold),
    )


class GSULayer(nn.Module):
    def __init__(self, cell_cls, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell_cls(*cell_args, **cell_kwargs)
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
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    ):
        super().__init__()
        cell_cls = get_cell_class(neuron_type)
        layer_kwargs = {
            "shared_weights": shared_weights,
            "bn": bn,
            "spike_threshold": float(spike_threshold),
        }
        layers = [GSULayer(cell_cls, input_size, hidden_size, **layer_kwargs)]
        for _ in range(num_layers - 1):
            layers.append(GSULayer(cell_cls, hidden_size, hidden_size, **layer_kwargs))
        self.layers = nn.ModuleList(layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neuron_type = normalize_neuron_type(neuron_type)
        self.spike_threshold = float(spike_threshold)

    def forward(self, input_seq: torch.Tensor, states: list[MemoryState]):
        output = input_seq
        new_states = []
        all_layer_outputs = [input_seq]
        for layer, state in zip(self.layers, states):
            output, new_state = layer(output, state)
            new_states.append(new_state)
            all_layer_outputs.append(output)
        return output, new_states, all_layer_outputs
