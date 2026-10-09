import os
import subprocess
import sys

from qlsgym.config import DEFAULT_PATH, constants, molecule_config
from qlsgym.molecules import h3o, thf
from qlsgym.physics.thermal import KB_OVER_HBAR
from qlsgym.spec import TWO_PI


def test_molecule_packs_take_their_settings_from_the_config():
    c = molecule_config("thf")
    m = thf.build()
    assert m.trap.n_nu == c["trap"]["n_nu"] and m.trap.eta == c["trap"]["eta"]
    assert m.window.tau_max_ms == c["window"]["tau_max_ms"] and m.window.n_tau == c["window"]["n_tau"]
    assert m.window.s_emb == c["embedding"]["s_emb"] and m.task.temperature_k == c["task"]["temperature_k"]
    assert len(m.primitives) == c["primitives"]["n"] and thf.J_MAX == c["tables"]["j_max"]
    assert h3o.build().trap.nu_f == TWO_PI * molecule_config("h3o")["trap"]["nu_f_khz"]
    assert KB_OVER_HBAR == TWO_PI * constants()["k_B_over_h_kHz_per_K"]


def test_environment_variable_points_at_another_config(tmp_path):
    alt = tmp_path / "physics.config"
    alt.write_text(DEFAULT_PATH.read_text().replace("n_nu = 8 ", "n_nu = 9 ", 1))
    code = "from qlsgym.molecules import thf; print(thf.build(n_primitives=0).trap.n_nu)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "QLSGYM_PHYSICS_CONFIG": str(alt)})
    assert out.stdout.strip() == "9"
