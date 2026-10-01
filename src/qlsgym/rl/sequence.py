"""Causal finite histories for sequence-aware quantum-control policies.

At decision time ``t`` the token history is

    H_t = ((s_i, a_{i-1}, k_{i-1}, b_i))_{i=t-K+1}^t,

where ``s_i`` is the post-measurement population belief, ``a_{i-1}`` is the
previous control, ``k_{i-1}`` is its measured branch, and
``b_i = (H-i)/H`` is the remaining pulse budget.  Histories are right aligned:
the current token is always at index ``K-1`` and leading positions are padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


SEQUENCE_ENCODERS = ("stack", "gru", "transformer")


def action_features(library) -> np.ndarray:
    """Return normalized ``(omega, tau, sigma, primitive)`` for every action.

    A final all-zero row represents START.  Frequencies are normalized by the
    Raman window and clipped only for off-window primitive pulses.
    """
    n_actions = library.n_actions
    features = np.zeros((n_actions + 1, 4), dtype=np.float32)
    window = library.molecule.window
    center = 0.5 * (window.omega_min + window.omega_max)
    half_width = max(0.5 * (window.omega_max - window.omega_min), 1e-12)
    tau_grid = library.molecule.tau_grid()
    tau_scale = max(float(tau_grid[-1]), 1e-12)
    for index in range(n_actions):
        action = library.decode(index)
        features[index] = (
            np.clip((float(action.omega) - center) / half_width, -2.0, 2.0),
            float(tau_grid[action.tau_index]) / tau_scale,
            1.0 if action.sigma == "+" else -1.0,
            float(action.is_primitive),
        )
    return features


@dataclass
class HistoryTensors:
    """Tensor representation of a batch of right-aligned causal histories."""

    state: torch.Tensor          # [B,K,n_states]
    previous_action: torch.Tensor  # [B,K], START = n_actions
    previous_outcome: torch.Tensor # [B,K], START = 2
    budget: torch.Tensor         # [B,K,1]
    valid: torch.Tensor          # [B,K]

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "state": self.state,
            "previous_action": self.previous_action,
            "previous_outcome": self.previous_outcome,
            "budget": self.budget,
            "valid": self.valid,
        }


class RollingHistory:
    """Mutable histories for a vectorized environment."""

    def __init__(self, initial_state: torch.Tensor, context_len: int,
                 n_actions: int, max_pulses: int):
        if context_len < 1:
            raise ValueError("context_len must be positive")
        state = initial_state.to(torch.float32)
        if state.ndim != 2:
            raise ValueError("initial_state must have shape [batch,n_states]")
        self.context_len = int(context_len)
        self.n_actions = int(n_actions)
        self.max_pulses = int(max_pulses)
        batch, n_states = state.shape
        device = state.device
        self.data = HistoryTensors(
            state=torch.zeros(batch, context_len, n_states, dtype=torch.float32, device=device),
            previous_action=torch.full(
                (batch, context_len), n_actions, dtype=torch.long, device=device
            ),
            previous_outcome=torch.full(
                (batch, context_len), 2, dtype=torch.long, device=device
            ),
            budget=torch.zeros(batch, context_len, 1, dtype=torch.float32, device=device),
            valid=torch.zeros(batch, context_len, dtype=torch.bool, device=device),
        )
        self.reset_rows(torch.ones(batch, dtype=torch.bool, device=device), state)

    def reset_rows(self, mask: torch.Tensor, state: torch.Tensor) -> None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.data.state.device)
        if mask.ndim != 1 or len(mask) != len(self.data.state):
            raise ValueError("mask must have one entry per history")
        if not bool(mask.any()):
            return
        current = state.to(device=self.data.state.device, dtype=torch.float32)
        if current.shape[0] == len(mask):
            current = current[mask]
        if current.shape != (int(mask.sum()), self.data.state.shape[-1]):
            raise ValueError("state rows do not match reset mask")
        self.data.state[mask] = 0
        self.data.previous_action[mask] = self.n_actions
        self.data.previous_outcome[mask] = 2
        self.data.budget[mask] = 0
        self.data.valid[mask] = False
        self.data.state[mask, -1] = current
        self.data.budget[mask, -1, 0] = 1.0
        self.data.valid[mask, -1] = True

    def append(self, next_state: torch.Tensor, action: torch.Tensor,
               outcome: torch.Tensor, steps: torch.Tensor) -> None:
        next_budget = (1.0 - steps.to(torch.float32) / self.max_pulses).clamp(0.0, 1.0)
        self.data = advance_history(
            self.data.as_dict(), next_state, action, outcome, next_budget
        )

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.data.as_dict().items()}

    def view(self) -> dict[str, torch.Tensor]:
        return self.data.as_dict()


def advance_history(history: dict[str, torch.Tensor], next_state: torch.Tensor,
                    action: torch.Tensor, outcome: torch.Tensor,
                    next_budget: torch.Tensor) -> HistoryTensors:
    """Append one token without mutating ``history``."""
    state = torch.cat((history["state"][:, 1:], next_state.to(torch.float32)[:, None]), dim=1)
    previous_action = torch.cat((
        history["previous_action"][:, 1:], action.to(torch.long)[:, None]
    ), dim=1)
    previous_outcome = torch.cat((
        history["previous_outcome"][:, 1:], outcome.to(torch.long)[:, None]
    ), dim=1)
    budget = torch.cat((
        history["budget"][:, 1:], next_budget.to(torch.float32)[:, None, None]
    ), dim=1)
    valid = torch.cat((
        history["valid"][:, 1:],
        torch.ones(len(next_state), 1, dtype=torch.bool, device=next_state.device),
    ), dim=1)
    return HistoryTensors(state, previous_action, previous_outcome, budget, valid)


def branch_histories(history: dict[str, torch.Tensor], next_state: torch.Tensor,
                     action: torch.Tensor, next_budget: torch.Tensor) -> dict[str, torch.Tensor]:
    """Construct the two counterfactual histories ``H_{t+1}^{(0/1)}``.

    ``next_state`` has shape ``[B,2,n_states]``.  The returned batch is ordered
    ``(row 0 branch 0, row 0 branch 1, row 1 branch 0, ...)``.
    """
    batch = next_state.shape[0]
    repeated = {
        key: value[:, None].expand(batch, 2, *value.shape[1:]).reshape(
            batch * 2, *value.shape[1:]
        )
        for key, value in history.items()
    }
    outcomes = torch.arange(2, device=next_state.device).repeat(batch)
    advanced = advance_history(
        repeated,
        next_state.reshape(batch * 2, -1),
        action[:, None].expand(batch, 2).reshape(-1),
        outcomes,
        next_budget[:, None].expand(batch, 2).reshape(-1),
    )
    return advanced.as_dict()


class HistoryEncoder(nn.Module):
    """Encode ``H_t`` with a frame stack, GRU, or causal Transformer."""

    def __init__(self, n_states: int, n_actions: int, context_len: int,
                 encoder: str, d_model: int, n_layers: int, n_heads: int,
                 ff_dim: int, dropout: float, obs: str,
                 physical_features: np.ndarray | torch.Tensor | None = None,
                 use_action: bool = True, use_outcome: bool = True,
                 use_budget: bool = True, use_physics: bool = True,
                 use_position: bool = True):
        super().__init__()
        if encoder not in SEQUENCE_ENCODERS:
            raise ValueError(f"encoder must be one of {SEQUENCE_ENCODERS}")
        if d_model < 1 or n_layers < 1 or context_len < 1:
            raise ValueError("d_model, n_layers and context_len must be positive")
        if encoder == "transformer" and d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.encoder_type = encoder
        self.context_len = int(context_len)
        self.obs = obs
        self.use_action = bool(use_action)
        self.use_outcome = bool(use_outcome)
        self.use_budget = bool(use_budget)
        self.use_physics = bool(use_physics)
        self.use_position = bool(use_position)
        self.state_projection = nn.Linear(n_states, d_model)
        self.action_embedding = nn.Embedding(n_actions + 1, d_model)
        self.outcome_embedding = nn.Embedding(3, d_model)
        self.budget_projection = nn.Linear(1, d_model)
        if physical_features is None:
            physical_features = torch.zeros(n_actions + 1, 4)
        self.register_buffer(
            "action_features", torch.as_tensor(physical_features, dtype=torch.float32)
        )
        self.physics_projection = nn.Linear(self.action_features.shape[1], d_model)
        self.position = nn.Parameter(torch.zeros(context_len, d_model))
        if encoder == "stack":
            self.sequence = nn.Sequential(
                nn.Linear(context_len * d_model, d_model), nn.Tanh()
            )
        elif encoder == "gru":
            self.sequence = nn.GRUCell(d_model, d_model)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_dim = int(d_model)

    def _tokens(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        state = history["state"].to(torch.float32).clamp_min(0.0)
        if self.obs == "sqrt":
            state = state.sqrt()
        elif self.obs != "p":
            raise ValueError("obs must be 'p' or 'sqrt'")
        token = self.state_projection(state)
        action = history["previous_action"].to(torch.long)
        if self.use_action:
            token = token + self.action_embedding(action)
        if self.use_outcome:
            token = token + self.outcome_embedding(history["previous_outcome"].to(torch.long))
        if self.use_budget:
            token = token + self.budget_projection(history["budget"].to(torch.float32))
        if self.use_physics:
            token = token + self.physics_projection(self.action_features[action])
        if self.use_position:
            token = token + self.position[None]
        return token * history["valid"].to(token.dtype).unsqueeze(-1)

    def forward(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        token = self._tokens(history)
        valid = history["valid"].to(torch.bool)
        if self.encoder_type == "stack":
            return self.sequence(token.flatten(1))
        if self.encoder_type == "gru":
            hidden = torch.zeros(
                len(token), self.output_dim, dtype=token.dtype, device=token.device
            )
            for index in range(self.context_len):
                update = self.sequence(token[:, index], hidden)
                hidden = torch.where(valid[:, index, None], update, hidden)
            return hidden
        # Histories are stored right aligned so the current token is always at
        # K-1.  Move the valid suffix to the left before causal attention.  If
        # right-aligned padding is passed directly, an early padded query has
        # every permitted key masked and softmax(-inf,...) produces NaN.
        length = valid.sum(dim=1)
        index = torch.arange(self.context_len, device=token.device)[None]
        order = (index + self.context_len - length[:, None]) % self.context_len
        token = token.gather(1, order[:, :, None].expand_as(token))
        valid = index < length[:, None]
        causal = torch.triu(
            torch.ones(self.context_len, self.context_len, dtype=torch.bool, device=token.device),
            diagonal=1,
        )
        encoded = self.sequence(token, mask=causal, src_key_padding_mask=~valid)
        batch = torch.arange(len(token), device=token.device)
        return encoded[batch, length - 1]


class SequenceHead(nn.Module):
    """Independent history encoder and one linear policy/value/Q head."""

    def __init__(self, n_states: int, n_actions: int, context_len: int,
                 encoder: str, d_model: int, n_layers: int, n_heads: int,
                 ff_dim: int, dropout: float, obs: str,
                 physical_features=None, output_gain: float = 1.0,
                 use_action: bool = True, use_outcome: bool = True,
                 use_budget: bool = True, use_physics: bool = True,
                 use_position: bool = True):
        super().__init__()
        self.encoder = HistoryEncoder(
            n_states, n_actions, context_len, encoder, d_model, n_layers,
            n_heads, ff_dim, dropout, obs, physical_features,
            use_action, use_outcome, use_budget, use_physics,
            use_position,
        )
        self.head = nn.Linear(d_model, n_actions)
        nn.init.orthogonal_(self.head.weight, output_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(history))
