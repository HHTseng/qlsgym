"""The qMDP-DQN baseline of arXiv:2608.03702 Appendix C.1, on qlsgym."""

from __future__ import annotations

from .agent import AgentConfig, QNetPolicy, epsilon, make_qnet, untrained_action_rows
from .anchor import FixedActionPolicy, best_fixed_action, one_step_yield
from .replay import Replay
from .train import (EVAL_SEED, load_policy, run_tag, select_best, snapshot_path,
                    stamp_snapshot_dir, train)

__all__ = ["AgentConfig", "QNetPolicy", "epsilon", "make_qnet", "untrained_action_rows",
           "FixedActionPolicy", "best_fixed_action", "one_step_yield", "Replay",
           "EVAL_SEED", "load_policy", "run_tag", "select_best", "snapshot_path",
           "stamp_snapshot_dir", "train", "DEVIATIONS"]


# Where this port departs from Appendix C.1 as printed.  Reported in the
# s06_rl payload under deviations.
DEVIATIONS = (
    {
        "id": "thz_actions_are_primitives",
        "appendix": "'the additional numerical pulses from the THz region were also "
                    "included in the action library'",
        "ours": "qlsgym models an off-window THz drive as one Primitive action at its own "
                "pi-time, not as a (sigma x omega x tau) sub-grid.  With H3O+'s 21 "
                "primitives the library is 35280 + 21 = 35301 actions, where FNO_REPL "
                "folded the same 21 resonances into the product grid and got "
                "35280 + 21*2*10 = 35700.",
        "consequence": "our action space is 399 actions smaller than FNO_REPL's; the two "
                       "convergence fractions are close relatives, not the same experiment.",
    },
    {
        "id": "evaluation_through_the_gym",
        "appendix": "'evaluated ... on 1000 Monte-Carlo trajectories'",
        "ours": "qlsgym.policies.rollout.rollout scores the agent, so it is measured by the "
                "same driver as every other policy in the gym.  It checks the Eq. 10 "
                "stopping rule *before* each pulse as well as after the last, so a belief "
                "already at p_target at t = 0 is a length-0 success; FNO_REPL's evaluator "
                "only checked after a step.",
        "consequence": "no difference for a 20 K thermal start (max_j p_j = 0.06); it matters "
                       "only for a molecule that starts converged.",
    },
    {
        "id": "one_configuration_not_twelve",
        "appendix": "Table 3: 12 sampled (tau_RL, eta_RL, rho, epsilon_end) configurations",
        "ours": "one configuration -- the paper's own dagger-marked row 3 "
                "(tau_RL = 1e-4, eta_RL = 5e-4, rho = 2, epsilon_end = 0.025).  The sweep "
                "machinery is deliberately not ported.",
        "consequence": "the paper's 0.428 is a maximum over 12 configurations x 30 snapshots "
                       "and ours is a maximum over the snapshots of one run, so ours carries "
                       "less selection optimism and is, if anything, the more conservative "
                       "estimate of the same quantity.",
    },
    {
        "id": "reward_is_not_in_the_paper",
        "appendix": "Appendix C.1 states no reward function",
        "ours": "the Ref. [29] (arXiv:2410.11839) reward that qlsgym implements: -1 per "
                "pulse minus rho when cos(p_t, p_t+1) > 1 - 1/n_states "
                "(EnvConfig(penalty_mode='indicator')).",
        "consequence": "a reading, not a quotation.  FNO_REPL measured the proportional "
                       "alternative and found it 3x worse, so this reading is not what "
                       "explains a shortfall against the paper.",
    },
    {
        "id": "epsilon_schedule_shape",
        "appendix": "'epsilon decayed from 1.0 to epsilon_end over 7.2e4 environment steps'",
        "ours": "a linear anneal that attains epsilon_end at 7.2e4 steps; Ref. [29]'s "
                "Eq. S16 is exponential, per episode, and never attains it.",
        "consequence": "FNO_REPL open question B3; the sentence in Appendix C.1 is the "
                       "reading implemented.",
    },
    {
        "id": "grid_size_is_a_knob",
        "appendix": "1764 frequencies x 2 polarisations x 10 durations = 35280 actions",
        "ours": "that is the default (ControlGrid.rl_discrete's own defaults), but the stage "
                "drops duration slots that the molecule's tau grid cannot hold (the synthetic "
                "one holds 5 of the 10), and cfg.overrides['n_freq'] / ['n_tau_slots'] shrink "
                "it further.",
        "consequence": "any payload whose library.n_grid is not 35280 is not the paper's "
                       "action space; the payload always reports the size it used.",
    },
)
