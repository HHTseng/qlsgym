"""Policies that hand difficult trajectories from a learned actor to a fallback."""

from __future__ import annotations

import copy

import numpy as np


class BatchedFallbackPolicy:
    """Use ``actor`` first, then latch individual rows onto ``fallback``.

    A row switches when its remaining pulse budget is small or its maximum
    belief has stopped setting a new best value.  The fallback is copied once
    per row so stateful controllers such as ``PhysicsEliminationPolicy`` keep
    private trajectory memory.  The object presents ``stateful = False`` on
    purpose: its ``act_batch`` method owns and resets that row-private state,
    allowing the evaluation driver to retain batched actor inference.
    """

    stateful = False

    def __init__(
        self,
        actor,
        fallback,
        max_pulses: int,
        switch_remaining: int = 0,
        stagnation_steps: int = 0,
        min_purity_gain: float = 1e-3,
        action_encoder=None,
    ):
        self.actor = actor
        self.fallback = fallback
        self.max_pulses = int(max_pulses)
        self.switch_remaining = int(switch_remaining)
        self.stagnation_steps = int(stagnation_steps)
        self.min_purity_gain = float(min_purity_gain)
        self.action_encoder = action_encoder
        if self.max_pulses < 1:
            raise ValueError("max_pulses must be positive")
        if not 0 <= self.switch_remaining <= self.max_pulses:
            raise ValueError("switch_remaining must lie in [0, max_pulses]")
        if self.stagnation_steps < 0 or self.min_purity_gain < 0:
            raise ValueError("stagnation_steps and min_purity_gain must be nonnegative")
        self.reset()

    def reset(self):
        self._row_fallbacks = []
        self._best_purity = None
        self._stagnation = None
        self._using_fallback = None

    def _initialize_rows(self, beliefs):
        rows = beliefs.shape[0]
        self._row_fallbacks = [copy.copy(self.fallback) for _ in range(rows)]
        for policy in self._row_fallbacks:
            if hasattr(policy, "reset"):
                policy.reset()
        self._best_purity = beliefs.max(axis=1).copy()
        self._stagnation = np.zeros(rows, dtype=np.int64)
        self._using_fallback = np.zeros(rows, dtype=bool)

    def act_batch(self, beliefs, t, rng):
        beliefs = np.asarray(beliefs, dtype=np.float64)
        if beliefs.ndim != 2:
            raise ValueError("beliefs must have shape (batch, states)")
        if self._best_purity is None:
            self._initialize_rows(beliefs)
        elif beliefs.shape[0] != self._best_purity.size:
            raise ValueError("batch size changed without reset")

        purity = beliefs.max(axis=1)
        improved = purity > self._best_purity + self.min_purity_gain
        self._best_purity = np.maximum(self._best_purity, purity)
        self._stagnation = np.where(improved, 0, self._stagnation + (int(t) > 0))

        trigger = np.zeros(beliefs.shape[0], dtype=bool)
        if self.switch_remaining:
            trigger |= self.max_pulses - int(t) <= self.switch_remaining
        if self.stagnation_steps:
            trigger |= self._stagnation >= self.stagnation_steps
        self._using_fallback |= trigger

        actor_actions = np.zeros(beliefs.shape[0], dtype=np.int64)
        actor_rows = np.flatnonzero(~self._using_fallback)
        if actor_rows.size:
            actor_actions[actor_rows] = np.asarray(
                self.actor.act_batch(beliefs[actor_rows], t, rng), dtype=np.int64,
            ).reshape(-1)

        fallback_rows = np.flatnonzero(self._using_fallback)
        if (fallback_rows.size and not getattr(self.fallback, "stateful", False)
                and hasattr(self.fallback, "act_batch")):
            actor_actions[fallback_rows] = np.asarray(
                self.fallback.act_batch(beliefs[fallback_rows], int(t), rng), dtype=np.int64,
            ).reshape(-1)
        else:
            for row in fallback_rows:
                pick = self._row_fallbacks[row].act(beliefs[row], int(t), rng)
                if pick is not None:
                    if isinstance(pick, (int, np.integer)):
                        actor_actions[row] = int(pick)
                    elif self.action_encoder is not None:
                        actor_actions[row] = int(self.action_encoder(pick))
                    else:
                        raise TypeError(
                            "fallback returned an action object but no action_encoder was provided"
                        )
        return actor_actions

    def act(self, belief, t, rng):
        belief = np.asarray(belief, dtype=np.float64)
        return int(self.act_batch(belief[None], t, rng)[0])


class ExactCandidateArbiterPolicy:
    """Admit a learned action only when exact one-step outcomes improve.

    The learned actor and baseline each propose one action.  The arbiter first
    compares immediate success probability, then expected posterior purity.
    It uses cached exact action tables and never advances or mutates the audit
    environment.
    """

    stateful = False

    def __init__(
        self,
        actor,
        baseline,
        exact_environment,
        success_margin: float = 0.0,
        purity_margin: float = 0.0,
    ):
        self.actor = actor
        self.baseline = baseline
        self.exact_environment = exact_environment
        self.success_margin = float(success_margin)
        self.purity_margin = float(purity_margin)
        if self.success_margin < 0 or self.purity_margin < 0:
            raise ValueError("arbiter margins must be nonnegative")

    def reset(self):
        if hasattr(self.actor, "reset"):
            self.actor.reset()
        if hasattr(self.baseline, "reset"):
            self.baseline.reset()

    def _scores(self, beliefs, actions):
        s0, s1, p0, p1, _, _, done0, done1 = self.exact_environment.branch_outcomes(
            beliefs, actions,
        )
        success = p0 * done0.to(p0.dtype) + p1 * done1.to(p1.dtype)
        purity = p0 * s0.max(-1).values + p1 * s1.max(-1).values
        return success.detach().cpu().numpy(), purity.detach().cpu().numpy()

    def act_batch(self, beliefs, t, rng):
        beliefs = np.asarray(beliefs, dtype=np.float64)
        learned = np.asarray(self.actor.act_batch(beliefs, t, rng), dtype=np.int64).reshape(-1)
        baseline = np.asarray(self.baseline.act_batch(beliefs, t, rng), dtype=np.int64).reshape(-1)
        learned_success, learned_purity = self._scores(beliefs, learned)
        base_success, base_purity = self._scores(beliefs, baseline)
        success_better = learned_success > base_success + self.success_margin
        success_tied = np.abs(learned_success - base_success) <= self.success_margin
        purity_better = learned_purity > base_purity + self.purity_margin
        use_learned = success_better | (success_tied & purity_better)
        return np.where(use_learned, learned, baseline)

    def act(self, belief, t, rng):
        belief = np.asarray(belief, dtype=np.float64)
        return int(self.act_batch(belief[None], t, rng)[0])
