"""Deep Q-Network (value-based) training with TorchRL."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment.custom_env import N_ACTIONS
from training.common import LOGS, CSVLogger, episode_returns, evaluate, make_env, mlp, save_policy

DQN_DEFAULT = dict(
    lr=5e-4, gamma=0.97, batch_size=128, buffer_size=50_000, hidden=128,
    eps_start=1.0, eps_end=0.05, eps_fraction=0.3,     # exploration schedule
    target_update="soft", tau=0.01, hard_interval=500,  # target network
    double_dqn=True, updates_per_batch=8, frames_per_batch=64,
    init_random_frames=2_000, loss="smooth_l1", grad_clip=10.0,
)


def train_dqn(cfg: dict | None = None, run: str = "dqn_default", total_frames: int = 300_000,
              seed: int = 0, verbose: bool = True) -> dict:
    cfg = {**DQN_DEFAULT, **(cfg or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(seed=seed)

    q_net = mlp(N_ACTIONS, cfg["hidden"], act=torch.nn.ReLU)
    q_actor = QValueActor(TensorDictModule(q_net, in_keys=["observation"], out_keys=["action_value"]),
                          in_keys=["observation"], spec=env.action_spec)
    greedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                           annealing_num_steps=int(cfg["eps_fraction"] * total_frames))
    policy = TensorDictSequential(q_actor, greedy)

    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames, init_random_frames=cfg["init_random_frames"])
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]),
                                batch_size=cfg["batch_size"])
    loss_mod = DQNLoss(q_actor, action_space=env.action_spec, delay_value=True,
                       double_dqn=cfg["double_dqn"], loss_function=cfg["loss"])
    loss_mod.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target_update"] == "soft":
        target = SoftUpdate(loss_mod, tau=cfg["tau"])
    else:
        target = HardUpdate(loss_mod, value_network_update_interval=cfg["hard_interval"])
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])

    log = CSVLogger(LOGS / "dqn" / f"{run}.csv")
    frames, recent, it = 0, [], 0
    for data in collector:
        frames += data.numel()
        rb.extend(data.reshape(-1))
        recent.extend(episode_returns(data).tolist())
        losses, qs = [], []
        if frames >= cfg["init_random_frames"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = rb.sample()
                loss = loss_mod(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["grad_clip"])
                optim.step()
                target.step()
                losses.append(loss.item())
            with torch.no_grad():
                qs.append(q_net(sample["observation"]).max(-1).values.mean().item())
        greedy.step(data.numel())
        it += 1
        if it % 20 == 0 and recent:
            row = dict(frames=frames, episodes_reward=float(np.mean(recent)), n_episodes=len(recent),
                       loss=float(np.mean(losses)) if losses else None,
                       q_mean=float(np.mean(qs)) if qs else None, epsilon=float(greedy.eps))
            log.log(**row)
            if verbose and it % 200 == 0:
                print(f"[DQN {run}] frames={frames:>7d} R={row['episodes_reward']:+8.2f} "
                      f"loss={row['loss'] or 0:.3f} eps={row['epsilon']:.3f}", flush=True)
            recent = []
    collector.shutdown()
    log.close()

    q_net.eval()
    summary = {"algo": "DQN", "run": run, "frames": frames,
               "eval_train": evaluate(q_net, "train"), "eval_stress": evaluate(q_net, "stress")}
    save_policy("dqn", run, q_net, cfg, summary)
    if verbose:
        print(json.dumps({k: summary[k] for k in ("run", "eval_train")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300_000)
    ap.add_argument("--run", default="dqn_default")
    ap.add_argument("--cfg", default="{}", help="JSON overrides, e.g. '{\"lr\": 1e-3}'")
    a = ap.parse_args()
    train_dqn(json.loads(a.cfg), a.run, a.frames)
