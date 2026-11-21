"""Report figures -> assets/*.png, plus markdown tables -> assets/tables.md."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.custom_env import ACTIONS, ImisoroComplianceEnv  # noqa: E402
from training.common import EVAL_SEEDS, LOGS, ROOT  # noqa: E402

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#2a6fdb", "reinforce": "#d1495b", "ppo": "#1b998b", "a2c": "#e8a33d",
         "random": "#9aa5b1", "wait": "#5c6b7a", "heuristic": "#7a5195"}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def smooth(y, k=10):
    y = pd.Series(y)
    return y.rolling(k, min_periods=1).mean().to_numpy()


def _curve(algo, run):
    return pd.read_csv(LOGS / algo / f"{run}.csv")


def baseline_eval():
    """Rule-based reference policies on the same evaluation seeds."""
    def heuristic(o, env):
        wtd = round(o[1] * 8)
        if wtd == 3:
            return 6
        if wtd == 2:
            return 3
        if wtd <= 1:
            return 1
        return 0

    rng = np.random.default_rng(0)
    pols = {"random": lambda o, e: int(rng.integers(len(ACTIONS))), "wait": lambda o, e: 0,
            "heuristic": heuristic}
    out = {}
    for name, pol in pols.items():
        for sc in ("train", "stress"):
            env = ImisoroComplianceEnv(sc)
            R, C, O = [], [], []
            for s in EVAL_SEEDS:
                o, _ = env.reset(seed=int(s))
                done, G = False, 0
                while not done:
                    o, r, t, tr, info = env.step(pol(o, env))
                    G += r
                    done = t or tr
                R.append(G)
                C.append(info["compliance_rate"])
                O.append(info["opted_out"])
            out[(name, sc)] = (float(np.mean(R)), float(np.std(R)), float(np.mean(C)), float(np.mean(O)))
    return out


def make_all():
    ASSETS.mkdir(exist_ok=True)
    df = pd.read_csv(LOGS / "sweep_results.csv")
    df0 = df[df.seed == 0]
    best = json.loads((ROOT / "models" / "best.json").read_text())["per_algo"]
    seeds = pd.read_csv(LOGS / "seed_robustness.csv") if (LOGS / "seed_robustness.csv").exists() else None

    # 1. comparative training reward (best run per algorithm)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for a in ALGOS:
        c = _curve(a, best[a]["run"])
        ax.plot(c.frames, smooth(c.episodes_reward, 15), color=COLOR[a], lw=1.8,
                label=f"{LABEL[a]} ({best[a]['run'].split('_')[1]})")
    ax.set(xlabel="Environment steps (weeks simulated)", ylabel="Mean episode return (smoothed)",
           title="Training reward of the best configuration per algorithm")
    ax.legend(frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig1_reward_comparison.png")
    plt.close(fig)

    # 2. sweep spread: all 12 runs per algorithm (convergence + stability)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        for r in df0[df0.algo == a].itertuples():
            c = _curve(a, r.run)
            isbest = r.run == best[a]["run"]
            ax.plot(c.frames, smooth(c.episodes_reward, 15), color=COLOR[a] if isbest else "#b8c2cc",
                    lw=2 if isbest else 0.8, zorder=3 if isbest else 1)
        ax.set_title(f"{LABEL[a]} (12 runs, best in colour)")
        ax.set_xlabel("steps")
    axes[0, 0].set_ylabel("episode return")
    axes[1, 0].set_ylabel("episode return")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig2_sweep_convergence.png")
    plt.close(fig)

    # 3. DQN objective: TD loss, mean max-Q, epsilon
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    show = [best["dqn"]["run"], "dqn_run01", "dqn_run04", "dqn_run05"]
    for run in dict.fromkeys(show):
        c = _curve("dqn", run).dropna(subset=["loss"])
        lab = df0.set_index("run").loc[run, "changed"]
        lab = "best" if run == best["dqn"]["run"] else lab
        axes[0].plot(c.frames, smooth(c.loss, 10), lw=1.2, label=lab)
        axes[1].plot(c.frames, smooth(c.q_mean, 10), lw=1.2)
        axes[2].plot(c.frames, c.epsilon, lw=1.2)
    axes[0].set(title="DQN TD loss (smooth L1)", xlabel="steps", yscale="log")
    axes[1].set(title="Mean max Q(s,a) on replay batch", xlabel="steps")
    axes[2].set(title="Exploration epsilon", xlabel="steps")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig3_dqn_objective.png")
    plt.close(fig)

    # 4. policy entropy for PG methods
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    for a in ["reinforce", "ppo", "a2c"]:
        c = _curve(a, best[a]["run"])
        axes[0].plot(c.frames, smooth(c.entropy, 10), color=COLOR[a], label=LABEL[a])
    axes[0].axhline(np.log(len(ACTIONS)), ls="--", c="grey", lw=0.8)
    axes[0].text(0, np.log(len(ACTIONS)) + 0.03, "uniform policy (ln 8)", fontsize=7, color="grey")
    axes[0].set(title="Policy entropy (best runs)", xlabel="steps", ylabel="nats")
    axes[0].legend(frameon=False)
    for run, lab in [("ppo_run10", "entropy 0.0"), ("ppo_run01", "entropy 0.01"), ("ppo_run11", "entropy 0.05")]:
        c = _curve("ppo", run)
        axes[1].plot(c.frames, smooth(c.entropy, 10), label=lab)
    axes[1].set(title="PPO entropy vs entropy coefficient", xlabel="steps")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig4_pg_entropy.png")
    plt.close(fig)

    # 5. generalisation: train vs stress scenario (+ rule-based references)
    base = baseline_eval()
    names = ALGOS + ["heuristic", "random", "wait"]
    tr, st, trc, stc, tro, sto = [], [], [], [], [], []
    for n in names:
        if n in ALGOS:
            r = df0.set_index("run").loc[best[n]["run"]]
            tr.append(r.eval_return_mean); st.append(r.stress_return_mean)
            trc.append(r.eval_compliance); stc.append(r.stress_compliance)
            tro.append(r.eval_opt_out); sto.append(r.stress_opt_out)
        else:
            tr.append(base[(n, "train")][0]); st.append(base[(n, "stress")][0])
            trc.append(base[(n, "train")][2]); stc.append(base[(n, "stress")][2])
            tro.append(base[(n, "train")][3]); sto.append(base[(n, "stress")][3])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, (A, B, t) in zip(axes, [(tr, st, "Mean return (200 unseen businesses)"),
                                   (trc, stc, "On-time filing rate"),
                                   (tro, sto, "Owner opt-out rate")]):
        ax.bar(x - 0.2, A, 0.4, label="train distribution", color=[COLOR[n] for n in names])
        ax.bar(x + 0.2, B, 0.4, label="stress (unseen, harder)", color=[COLOR[n] for n in names],
               alpha=0.45, hatch="//", edgecolor="white")
        ax.set_xticks(x, [LABEL.get(n, n) for n in names], rotation=35, ha="right", fontsize=8)
        ax.set_title(t)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].axhline(0, c="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig5_generalization.png")
    plt.close(fig)

    # 6. learned action distributions
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    w = 0.2
    for i, a in enumerate(ALGOS):
        dist = json.loads(df0.set_index("run").loc[best[a]["run"], "action_dist"])
        ax.bar(np.arange(len(ACTIONS)) + (i - 1.5) * w, dist, w, color=COLOR[a], label=LABEL[a])
    ax.set_xticks(np.arange(len(ACTIONS)), [s.replace("_", "\n") for s in ACTIONS], fontsize=7)
    ax.set(ylabel="share of weeks", title="How often each best agent uses each intervention (greedy policy)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig6_action_distribution.png")
    plt.close(fig)

    # tables
    lines = []
    for a in ALGOS:
        g = df0[df0.algo == a].sort_values("idx")
        hp_cols = [c for c in g.columns if c.startswith("hp_") and g[c].notna().any()]
        lines.append(f"\n### {LABEL[a]}\n")
        lines.append("| Run | Changed vs default | Return (mean±sd) | On-time % | Stress return | Opt-out % |")
        lines.append("|---|---|---|---|---|---|")
        for r in g.itertuples():
            lines.append(f"| {r.idx} | {r.changed} | {r.eval_return_mean:+.1f} ± {r.eval_return_std:.1f} | "
                         f"{100 * r.eval_compliance:.1f} | {r.stress_return_mean:+.1f} | {100 * r.eval_opt_out:.1f} |")
        lines.append("\nDefaults: " + ", ".join(f"{c[3:]}={g.iloc[0][c]}" for c in hp_cols))
    if seeds is not None:
        lines.append("\n### Seed robustness (best config, seeds 0-2)\n")
        lines.append("| Algorithm | Run | Return mean over seeds | sd across seeds | On-time % |")
        lines.append("|---|---|---|---|---|")
        for a in ALGOS:
            s0 = df0.set_index("run").loc[best[a]["run"]]
            vals = [s0.eval_return_mean] + seeds[seeds.algo == a].eval_return_mean.tolist()
            comps = [s0.eval_compliance] + seeds[seeds.algo == a].eval_compliance.tolist()
            lines.append(f"| {LABEL[a]} | {best[a]['run']} | {np.mean(vals):+.1f} | {np.std(vals):.1f} | "
                         f"{100 * np.mean(comps):.1f} |")
    lines.append("\n### Rule-based references (same 200 evaluation businesses)\n")
    for n in ["heuristic", "random", "wait"]:
        lines.append(f"- {n}: train {base[(n, 'train')][0]:+.1f} (on-time {100 * base[(n, 'train')][2]:.1f}%), "
                     f"stress {base[(n, 'stress')][0]:+.1f} (on-time {100 * base[(n, 'stress')][2]:.1f}%, "
                     f"opt-out {100 * base[(n, 'stress')][3]:.1f}%)")
    (ASSETS / "tables.md").write_text("\n".join(lines))
    print(f"figures + tables written to {ASSETS}")


if __name__ == "__main__":
    make_all()
