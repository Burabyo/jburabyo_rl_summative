"""Imisoro Compliance Coach - entry point.

    uv run main.py                         # play the best trained agent (3D view + verbose terminal)
    uv run main.py play --algo ppo         # play a specific algorithm's best model
    uv run main.py random                  # random agent, to inspect the environment
    uv run main.py train --algo dqn        # train one algorithm with default hyperparameters
    uv run main.py sweep                   # 12 hyperparameter runs x 4 algorithms (+ seed checks)
    uv run main.py plots                   # regenerate report figures into assets/
    uv run main.py serve                   # JSON API: POST /api/recommend {"observation": [...]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _best_model(algo: str | None) -> tuple[str, Path]:
    best = json.loads((ROOT / "models" / "best.json").read_text())
    algo = algo or best["overall"]
    return algo, ROOT / best["per_algo"][algo]["path"]


def _recommender(net):
    import torch

    from environment.custom_env import ACTIONS

    def rec(obs):
        with torch.no_grad():
            scores = net(torch.as_tensor([obs], dtype=torch.float32))[0]
        a = int(scores.argmax())
        return {"action_id": a, "action": ACTIONS[a], "scores": [round(float(s), 4) for s in scores]}
    return rec


def play(algo: str | None, episodes: int, fps: float, scenario: str, seed: int,
         browser: bool, random_agent: bool = False, hold: bool = True):
    from environment.custom_env import ACTIONS, ImisoroComplianceEnv
    from environment.rendering import LiveRenderer

    renderer = LiveRenderer.get(open_browser=browser)
    if random_agent:
        policy, name = None, "RANDOM"
    else:
        from training.common import load_policy
        algo, path = _best_model(algo)
        net = load_policy(path)
        rec = _recommender(net)
        renderer.recommender = rec
        policy, name = rec, f"{algo.upper()} ({path.relative_to(ROOT)})"
    print(f"\n=== Imisoro Compliance Coach | agent: {name} | scenario: {scenario} ===")
    if browser:
        time.sleep(2.5)  # give the browser time to load the scene
    env = ImisoroComplianceEnv(scenario=scenario, render_mode="human")
    rng = np.random.default_rng(seed)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        print(f"\n--- Episode {ep + 1}: {['FLAT', 'LUMP_SUM', 'REAL'][env.regime]} regime, "
              f"VAT={env.vat_registered}, employees={env.has_employees}, "
              f"{sum(len(v) for v in env.calendar.values())} deadlines this year ---")
        done = False
        while not done:
            a = int(rng.integers(len(ACTIONS))) if policy is None else policy(obs.tolist())["action_id"]
            obs, r, term, trunc, info = env.step(a)
            print(env.ansi_line(), flush=True)
            done = term or trunc
            time.sleep(1.0 / fps)
        results.append((env.ep_return, info["compliance_rate"], info["opted_out"]))
        print(f"--- Episode {ep + 1} done: return={env.ep_return:+.1f} compliance={info['compliance_rate']:.0%} "
              f"filed={info['filed']} missed={info['missed']} underpaid={info['underpaid']} "
              f"opted_out={info['opted_out']} ---")
        time.sleep(2.0)
    R = np.array(results, dtype=float)
    print(f"\nSummary over {episodes} episode(s): mean return {R[:, 0].mean():+.1f}, "
          f"mean compliance {R[:, 1].mean():.0%}, opt-outs {int(R[:, 2].sum())}")
    if hold and browser:
        print(f"3D view still live at {renderer.url}  (Ctrl+C to exit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def serve(algo: str | None):
    from environment.rendering import LiveRenderer
    from training.common import load_policy
    algo, path = _best_model(algo)
    r = LiveRenderer.get(open_browser=False)
    r.recommender = _recommender(load_policy(path))
    print(f"Serving {algo.upper()} policy: POST {r.url}api/recommend  body={{\"observation\": [21 floats]}}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("play", "random"):
        p = sub.add_parser(name)
        p.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], default=None)
        p.add_argument("--episodes", type=int, default=2)
        p.add_argument("--fps", type=float, default=3.0)
        p.add_argument("--scenario", choices=["train", "stress"], default="train")
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--no-browser", action="store_true")
        p.add_argument("--no-hold", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], required=True)
    t.add_argument("--frames", type=int, default=300_000)
    s = sub.add_parser("sweep")
    s.add_argument("--algos", nargs="+", default=["dqn", "reinforce", "ppo", "a2c"])
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--frames", type=int, default=300_000)
    sub.add_parser("plots")
    v = sub.add_parser("serve")
    v.add_argument("--algo", default=None)
    a = ap.parse_args()

    cmd = a.cmd or "play"
    if cmd in ("play", "random"):
        d = dict(algo=None, episodes=2, fps=3.0, scenario="train", seed=2026, no_browser=False, no_hold=False)
        d.update({k: v for k, v in vars(a).items() if k in d})
        play(d["algo"], d["episodes"], d["fps"], d["scenario"], d["seed"], not d["no_browser"],
             random_agent=(cmd == "random"), hold=not d["no_hold"])
    elif cmd == "train":
        if a.algo == "dqn":
            from training.dqn_training import train_dqn
            train_dqn(total_frames=a.frames)
        else:
            from training.pg_training import train_pg
            train_pg(a.algo, total_frames=a.frames)
    elif cmd == "sweep":
        from training.sweep import run_sweep
        run_sweep(a.algos, a.workers, a.frames)
    elif cmd == "plots":
        from training.plots import make_all
        make_all()
    elif cmd == "serve":
        serve(a.algo)


if __name__ == "__main__":
    main()
