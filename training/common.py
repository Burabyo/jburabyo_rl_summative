"""Shared helpers: TorchRL env factory, networks, evaluation, logging, model I/O."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import GymWrapper, RewardSum, StepCounter, TransformedEnv

from environment.custom_env import N_ACTIONS, ImisoroComplianceEnv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
OBS_DIM = len(ImisoroComplianceEnv.OBS_NAMES)
EVAL_SEEDS = range(900_000, 900_200)  # never used during training


def make_env(scenario: str = "train", seed: int = 0) -> TransformedEnv:
    """The single environment every algorithm trains on (one-hot action spec)."""
    env = GymWrapper(ImisoroComplianceEnv(scenario=scenario))
    env = TransformedEnv(env)
    env.append_transform(StepCounter())
    env.append_transform(RewardSum())
    env.set_seed(seed)
    return env


def mlp(out_dim: int, hidden: int, layers: int = 2, act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = OBS_DIM
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


@torch.no_grad()
def evaluate(net: nn.Module, scenario: str = "train", seeds=EVAL_SEEDS) -> dict:
    """Greedy rollout of a network mapping observation -> action scores (Q-values or logits)."""
    env = ImisoroComplianceEnv(scenario=scenario)
    rets, comp, opt, filed, missed = [], [], [], [], []
    actions = np.zeros(N_ACTIONS)
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, G = False, 0.0
        while not done:
            a = int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
            actions[a] += 1
            obs, r, term, trunc, info = env.step(a)
            G += r
            done = term or trunc
        rets.append(G)
        comp.append(info["compliance_rate"])
        opt.append(info["opted_out"])
        filed.append(info["filed"])
        missed.append(info["missed"])
    return {
        "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
        "compliance": float(np.mean(comp)), "opt_out": float(np.mean(opt)),
        "filed": float(np.mean(filed)), "missed": float(np.mean(missed)),
        "action_dist": (actions / actions.sum()).round(4).tolist(),
    }


class CSVLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = None
        self.t0 = time.time()

    def log(self, **row):
        row = {"time_s": round(time.time() - self.t0, 2), **row}
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            self.w.writeheader()
        self.w.writerow({k: row.get(k) for k in self.w.fieldnames})
        self.f.flush()

    def close(self):
        self.f.close()


def episode_returns(data) -> np.ndarray:
    """Returns of episodes that finished inside a collected batch."""
    done = data["next", "done"].squeeze(-1)
    return data["next", "episode_reward"].squeeze(-1)[done].cpu().numpy()


def save_policy(algo_dir: str, run: str, net: nn.Module, cfg: dict, summary: dict):
    """Save weights (state_dict), a TorchScript export for serving, and metadata."""
    d = MODELS / algo_dir
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "cfg": cfg}, d / f"{run}.pt")
    scripted = torch.jit.trace(net.eval(), torch.zeros(1, OBS_DIM))
    scripted.save(str(d / f"{run}.ts"))
    (d / f"{run}.json").write_text(json.dumps({"cfg": cfg, **summary}, indent=2))


def load_policy(path: str | Path) -> nn.Module:
    """Load a TorchScript policy (obs[1, OBS_DIM] -> action scores)."""
    return torch.jit.load(str(Path(path).with_suffix(".ts"))).eval()
