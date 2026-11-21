"""Hyperparameter sweeps: 12 configurations per algorithm on the same environment and
the same frame budget, run in parallel processes. Results -> logs/sweep_results.csv."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from training.common import LOGS, MODELS

FRAMES = 300_000

SWEEPS: dict[str, list[dict]] = {
    "dqn": [
        {},
        {"lr": 1e-4},
        {"lr": 2e-3},
        {"gamma": 0.90},
        {"gamma": 0.995},
        {"eps_fraction": 0.1, "eps_end": 0.01},
        {"eps_fraction": 0.6, "eps_end": 0.10},
        {"target_update": "hard", "hard_interval": 200},
        {"target_update": "hard", "hard_interval": 2000},
        {"double_dqn": False},
        {"buffer_size": 5_000, "batch_size": 32},
        {"hidden": 256, "batch_size": 256, "updates_per_batch": 16},
    ],
    "reinforce": [
        {},
        {"baseline": "none", "normalize_returns": False},
        {"baseline": "mean"},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.90},
        {"gamma": 0.995},
        {"episodes_per_batch": 16, "n_envs": 16},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"hidden": 128},
        {"normalize_returns": False},
    ],
    "ppo": [
        {},
        {"lr": 1e-4},
        {"lr": 1e-3},
        {"clip_epsilon": 0.1},
        {"clip_epsilon": 0.3},
        {"epochs": 10},
        {"epochs": 2, "minibatch_size": 1024},
        {"gae_lambda": 0.80},
        {"gamma": 0.995},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"steps_per_env": 32, "minibatch_size": 64},
    ],
    "a2c": [
        {},
        {"lr": 2.5e-4},
        {"lr": 2e-3},
        {"steps_per_env": 5},
        {"steps_per_env": 32},
        {"gae_lambda": 0.95},
        {"gamma": 0.90},
        {"gamma": 0.995},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"critic_coeff": 1.0, "normalize_advantage": True},
        {"n_envs": 16, "hidden": 128},
    ],
}


def _job(algo: str, idx: int, overrides: dict, seed: int = 0, frames: int = FRAMES) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    run = f"{algo}_run{idx:02d}" + (f"_seed{seed}" if seed else "")
    saved = MODELS / ("dqn" if algo == "dqn" else "pg") / f"{run}.json"
    if saved.exists() and os.environ.get("SWEEP_FORCE") != "1":  # resume: reuse finished runs
        s = json.loads(saved.read_text())
        cfg = s["cfg"]
        cfg.pop("algo", None)
    elif algo == "dqn":
        from training.dqn_training import DQN_DEFAULT, train_dqn
        s = train_dqn(overrides, run, frames, seed=seed, verbose=False)
        cfg = {**DQN_DEFAULT, **overrides}
    else:
        from training.pg_training import DEFAULTS, train_pg
        s = train_pg(algo, overrides, run, frames, seed=seed, verbose=False)
        cfg = {**DEFAULTS[algo], **overrides}
    row = {"algo": algo, "run": run, "idx": idx, "seed": seed,
           "changed": json.dumps(overrides) if overrides else "default"}
    row.update({f"eval_{k}": v for k, v in s["eval_train"].items() if k != "action_dist"})
    row.update({f"stress_{k}": v for k, v in s["eval_stress"].items() if k != "action_dist"})
    row["action_dist"] = json.dumps(s["eval_train"]["action_dist"])
    row.update({f"hp_{k}": v for k, v in cfg.items()})
    print(f"done {run}: R={row['eval_return_mean']:+.1f} comp={row['eval_compliance']:.3f}", flush=True)
    return row


def run_sweep(algos=("dqn", "reinforce", "ppo", "a2c"), workers: int = 8, frames: int = FRAMES,
              robustness_seeds=(1, 2)):
    jobs = [(a, i, o) for a in algos for i, o in enumerate(SWEEPS[a], start=1)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, a, i, o, 0, frames) for a, i, o in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    df = pd.DataFrame(rows).sort_values(["algo", "idx"])
    out = LOGS / "sweep_results.csv"
    if out.exists():  # merge with algos not re-run this time
        old = pd.read_csv(out)
        df = pd.concat([old[~old.algo.isin(algos)], df]).sort_values(["algo", "idx"])
    df.to_csv(out, index=False)

    # re-train each algorithm's best configuration with extra seeds (robustness check)
    best = df[df.seed == 0].loc[lambda d: d.groupby("algo").eval_return_mean.idxmax()]
    seed_rows = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, r.algo, int(r.idx), SWEEPS[r.algo][int(r.idx) - 1], s, frames)
                for r in best.itertuples() for s in robustness_seeds]
        for f in as_completed(futs):
            seed_rows.append(f.result())
    pd.DataFrame(seed_rows).sort_values(["algo", "seed"]).to_csv(LOGS / "seed_robustness.csv", index=False)
    select_best(df)
    return df


def select_best(df: pd.DataFrame | None = None):
    """Copy each algorithm's best seed-0 run to models/<dir>/<algo>_best.* and record the overall best."""
    df = df if df is not None else pd.read_csv(LOGS / "sweep_results.csv")
    df = df[df.seed == 0]
    summary = {}
    for algo, g in df.groupby("algo"):
        r = g.loc[g.eval_return_mean.idxmax()]
        d = "dqn" if algo == "dqn" else "pg"
        for ext in (".pt", ".ts", ".json"):
            shutil.copy(MODELS / d / f"{r.run}{ext}", MODELS / d / f"{algo}_best{ext}")
        summary[algo] = {"run": r.run, "path": f"models/{d}/{algo}_best.ts",
                         "eval_return_mean": float(r.eval_return_mean),
                         "eval_compliance": float(r.eval_compliance)}
    overall = max(summary, key=lambda a: summary[a]["eval_return_mean"])
    (MODELS / "best.json").write_text(json.dumps({"overall": overall, "per_algo": summary}, indent=2))
    print(json.dumps({"overall": overall, "per_algo": summary}, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["dqn", "reinforce", "ppo", "a2c"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frames", type=int, default=FRAMES)
    a = ap.parse_args()
    run_sweep(a.algos, a.workers, a.frames)
