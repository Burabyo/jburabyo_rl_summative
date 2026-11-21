import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTIONS, ImisoroComplianceEnv


def test_gymnasium_api():
    check_env(ImisoroComplianceEnv(), skip_render_check=True)


def test_episode_is_one_tax_year_and_obs_in_bounds():
    env = ImisoroComplianceEnv()
    obs, _ = env.reset(seed=1)
    steps, done = 0, False
    while not done:
        assert env.observation_space.contains(obs)
        obs, r, term, trunc, info = env.step(0)
        steps += 1
        done = term or trunc
    assert steps == 52 and trunc and not term


def test_overcontact_causes_opt_out():
    env = ImisoroComplianceEnv("stress")
    outs = 0
    for s in range(20):
        env.reset(seed=s)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(ACTIONS.index("EDUCATION_MODULE"))
            done = term or trunc
        outs += info["opted_out"]
    assert outs >= 15


def test_actions_change_persistent_state():
    env = ImisoroComplianceEnv()
    env.reset(seed=3)
    env.trust, env.fatigue = 1.0, 0.0
    k0 = env.knowledge
    for _ in range(3):
        env.step(ACTIONS.index("EDUCATION_MODULE"))
        env.fatigue = 0.0
    assert env.knowledge > k0


def test_seeded_reset_is_deterministic():
    a, b = ImisoroComplianceEnv(), ImisoroComplianceEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.allclose(oa, ob) and a.calendar == b.calendar
