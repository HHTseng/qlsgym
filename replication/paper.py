"""Every number arXiv:2608.03702 states: a quotation, never our own measurement."""

from __future__ import annotations

# Fig. 3 -- forward-propagation accuracy  (measured by s04_accuracy)

# Sec. III.1: "a representative trajectory at omega/2pi = 5159.6 kHz".
FIG3A_OMEGA_OVER_2PI_KHZ = 5159.6

# Sec. III.1: "the population infidelity computed over all basis states and
# averaged over time ... lies low at 1.03e-3".  A *full-space* (444 molecular
# states, 888 channels) number; ours is per block, so the comparison is
# block-by-block and is labelled as such.
FIG3A_TIME_AVG_INFIDELITY = 1.03e-3

# Sec. III.1: "the mean relative error (MRE) over the plotted states ...
# remains below 3.5% throughout the evolution".  Undefined as written where
# the reference population vanishes; surrogate.metrics.mean_relative_error
# restricts it to the channels that actually move (its active_threshold).
FIG3A_MRE_BOUND = 0.035

# Sec. III.1 / Fig. 3b: "The median Ibar_p(tau) across test frequencies stays
# close to 1e-4, and the 95th percentile remains below 2e-3."  Both are over
# *uniformly sampled* test frequencies, where 94-99 % of the window is off
# every resonance (s01_physics measures the fraction) -- which is why R7
# requires the same number stratified and as a ratio to the trivial predictor.
FIG3B_MEDIAN = 1.0e-4
FIG3B_P95_BOUND = 2.0e-3

# Sec. III.1 / Fig. 3c: "The time-averaged infidelity is below 3e-3 for most
# test frequencies."
FIG3C_TYPICAL_BOUND = 3.0e-3

# Fig. 3b caption: L_init = 500 random mixed initial states per test
# frequency (Config.n_test_init).
FIG3B_L_INIT = 500

# Fig. 5 -- inverse design  (measured by s05_planner, s06_rl)

# Sec. III.3.  Convergence is a *fraction* of Monte-Carlo rollouts that reach
# P_TARGET within MAX_PULSES; pulse counts are means over all
# N_MC_ROLLOUTS rollouts.  None marks a quantity the paper plots
# but never states numerically (the Delta omega = 2.5 group, and every
# "average over generated sequences" marker).  The groups are the paper's
# frequency-refinement settings; discrete is the fixed grid, which is the
# only one s05_planner reproduces -- the others are quoted, not matched.
FIG5A_PAPER: dict[str, dict[str, dict[str, float | None]]] = {
    "dw0.25": {
        "best": {"convergence": 0.850, "mean_pulses": 27.51},
        "refined": {"convergence": 0.862, "mean_pulses": 26.54},
    },
    "dw1.0": {
        "best": {"convergence": 0.829, "mean_pulses": 30.09},
        "refined": {"convergence": 0.856, "mean_pulses": 28.34},
    },
    "dw2.5": {
        "best": {"convergence": None, "mean_pulses": None},
        "refined": {"convergence": None, "mean_pulses": None},
    },
    "discrete": {
        "best": {"convergence": 0.794, "mean_pulses": 27.9},
        "refined": {"convergence": None, "mean_pulses": None},
    },
}

# Fig. 5a group order and display labels, left to right as published.
FIG5A_GROUPS: tuple[tuple[str, str], ...] = (
    ("dw0.25", r"$\Delta\omega=0.25$"),
    ("dw1.0", r"$\Delta\omega=1.0$"),
    ("dw2.5", r"$\Delta\omega=2.5$"),
    ("discrete", "Discrete\ngrid"),
)

# Sec. III.3 / Appendix C.1: RL's "42.8% convergence and larger mean pulse
# count of 49.0"; Appendix C.1 gives 0.428 +- 0.016 and 48.974.
FIG5A_RL_PAPER: dict[str, float] = {
    "convergence": 0.428,
    "convergence_err": 0.016,
    "mean_pulses": 48.974,
}

# Sec. III.3: "approximately 7.5 hours of training across the 12
# hyperparameter settings and an additional ~2.5 hours to validate saved
# policy snapshots", against "approximately 10-20 minutes" for FNO-SPMP.
# Fig. 5b's abscissa runs to ~35 min, i.e. it is *per configuration*
# (7.5 h / 12 = 37.5 min), not the whole sweep.
RL_SWEEP_TRAIN_HOURS = 7.5
RL_SNAPSHOT_EVAL_HOURS = 2.5
RL_N_CONFIGS = 12
FNO_SPMP_MINUTES: tuple[float, float] = (10.0, 20.0)

# Appendix C.1: the daggered row of Table 3 (the configuration and snapshot
# whose numbers FIG5A_RL_PAPER reports).
RL_SELECTED_CONFIG = 3
RL_SELECTED_EPISODE = 2900

# Protocol constants (Sec. II.4 / Sec. III.3)

# The stopping rule and budget every arm of Fig. 5 shares.  qlsgym carries
# the same two on Molecule.task; s01_physics checks they agree, so these
# are the quotation and molecule.task is the measurement.
P_TARGET = 0.98
MAX_PULSES = 80

# Sec. III.3: the paper averages over 1000 Monte-Carlo rollouts
# (Config.n_rollouts is ours, and is smaller by default).
N_MC_ROLLOUTS = 1000
