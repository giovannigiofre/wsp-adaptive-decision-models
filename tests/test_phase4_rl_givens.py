"""Step 4.9b: the two additions to phase3.rl (harvest_steps in the run name, the givens control objective)."""
import yaml

from phase3.objectives import KIND
from phase3.rl import run_name


def _rc():
    return yaml.safe_load(open("configs/phase4_rl_givens.yaml"))["rl"]


def test_run_names_of_the_three_arms_differ_and_default_is_unchanged():
    rc = _rc()
    assert run_name(rc, 0) == "self_diffusion_seed0"            # same name as the 4.4 arm: harvest_steps unset
    assert run_name({**rc, "harvest_steps": 16}, 1) == "self_diffusion_h16_seed1"
    assert run_name({**rc, "objective": "givens"}, 2) == "givens_seed2"


def test_givens_control_uses_the_diffusion_branch():
    assert KIND["givens"] == "diffusion"


def test_config_differs_from_the_44_config_only_in_start_and_objective():
    a, b = (yaml.safe_load(open(f)) for f in ("configs/phase4_rl.yaml", "configs/phase4_rl_givens.yaml"))
    assert a["init"] != b["init"] and b["init"]["checkpoint"] == "4.8e/givens_long/final.pt"
    ra, rb = a["rl"], b["rl"]
    assert rb.pop("harvest_steps") is None and rb["objective_defaults"].pop("givens") == {}
    assert ra.pop("objective") == "wd1" and rb.pop("objective") == "self_diffusion"
    assert ra == rb and {k: v for k, v in a.items() if k not in ("init", "rl")} == {k: v for k, v in b.items() if k not in ("init", "rl")}
