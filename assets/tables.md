
### DQN

| Run | Changed vs default | Return (mean±sd) | On-time % | Stress return | Opt-out % |
|---|---|---|---|---|---|
| 1 | default | -30.8 ± 56.4 | 67.7 | -59.6 | 0.0 |
| 2 | {"lr": 0.0001} | -31.0 ± 52.0 | 66.2 | -96.8 | 0.0 |
| 3 | {"lr": 0.002} | -24.0 ± 56.2 | 68.5 | -83.7 | 0.0 |
| 4 | {"gamma": 0.9} | -26.4 ± 52.0 | 68.5 | -102.9 | 0.0 |
| 5 | {"gamma": 0.995} | -53.6 ± 55.2 | 63.9 | -78.7 | 0.0 |
| 6 | {"eps_fraction": 0.1, "eps_end": 0.01} | -48.9 ± 66.1 | 62.8 | -114.8 | 0.0 |
| 7 | {"eps_fraction": 0.6, "eps_end": 0.1} | -31.9 ± 61.4 | 68.4 | -84.0 | 0.0 |
| 8 | {"target_update": "hard", "hard_interval": 200} | -19.8 ± 54.2 | 67.6 | -55.6 | 0.0 |
| 9 | {"target_update": "hard", "hard_interval": 2000} | -30.7 ± 65.5 | 67.1 | -100.7 | 0.0 |
| 10 | {"double_dqn": false} | -44.3 ± 59.9 | 58.8 | -77.4 | 0.0 |
| 11 | {"buffer_size": 5000, "batch_size": 32} | -24.5 ± 46.7 | 71.8 | -49.2 | 0.0 |
| 12 | {"hidden": 256, "batch_size": 256, "updates_per_batch": 16} | -35.1 ± 64.2 | 66.0 | -93.7 | 0.0 |

Defaults: lr=0.0005, gamma=0.97, batch_size=128.0, buffer_size=50000.0, hidden=128, eps_start=1.0, eps_end=0.05, eps_fraction=0.3, target_update=soft, tau=0.01, hard_interval=500.0, double_dqn=True, updates_per_batch=8.0, frames_per_batch=64.0, init_random_frames=2000.0, loss=smooth_l1, grad_clip=10.0

### REINFORCE

| Run | Changed vs default | Return (mean±sd) | On-time % | Stress return | Opt-out % |
|---|---|---|---|---|---|
| 1 | default | -18.3 ± 39.1 | 69.2 | -42.5 | 0.0 |
| 2 | {"baseline": "none", "normalize_returns": false} | -19.4 ± 47.3 | 61.4 | -43.3 | 0.0 |
| 3 | {"baseline": "mean"} | -26.0 ± 51.4 | 64.1 | -42.9 | 0.0 |
| 4 | {"lr": 0.0003} | -30.4 ± 38.0 | 70.8 | -52.2 | 0.0 |
| 5 | {"lr": 0.003} | -20.3 ± 39.3 | 72.6 | -46.3 | 0.0 |
| 6 | {"gamma": 0.9} | -48.8 ± 62.9 | 61.1 | -120.0 | 0.0 |
| 7 | {"gamma": 0.995} | -22.3 ± 40.0 | 69.6 | -48.0 | 0.0 |
| 8 | {"episodes_per_batch": 16, "n_envs": 16} | -16.9 ± 41.5 | 73.5 | -42.9 | 0.0 |
| 9 | {"entropy_coeff": 0.0} | -21.3 ± 35.8 | 68.4 | -42.5 | 0.0 |
| 10 | {"entropy_coeff": 0.05} | -22.3 ± 37.4 | 70.8 | -42.7 | 0.0 |
| 11 | {"hidden": 128} | -16.6 ± 55.2 | 65.2 | -62.1 | 0.0 |
| 12 | {"normalize_returns": false} | -21.1 ± 37.7 | 67.8 | -43.0 | 0.0 |

Defaults: lr=0.001, gamma=0.97, hidden=64, grad_clip=1.0, episodes_per_batch=8.0, n_envs=8.0, entropy_coeff=0.01, baseline=value, normalize_returns=True, critic_coeff=0.5

### PPO

| Run | Changed vs default | Return (mean±sd) | On-time % | Stress return | Opt-out % |
|---|---|---|---|---|---|
| 1 | default | -9.9 ± 39.2 | 72.5 | -43.6 | 0.0 |
| 2 | {"lr": 0.0001} | -51.2 ± 65.0 | 58.6 | -118.5 | 0.0 |
| 3 | {"lr": 0.001} | -25.4 ± 77.9 | 73.7 | -97.6 | 0.0 |
| 4 | {"clip_epsilon": 0.1} | -13.5 ± 41.6 | 72.5 | -44.0 | 0.0 |
| 5 | {"clip_epsilon": 0.3} | -16.0 ± 38.4 | 75.5 | -45.1 | 0.0 |
| 6 | {"epochs": 10} | -6.2 ± 50.1 | 74.8 | -44.1 | 0.0 |
| 7 | {"epochs": 2, "minibatch_size": 1024} | -20.5 ± 43.1 | 64.7 | -45.2 | 0.0 |
| 8 | {"gae_lambda": 0.8} | -41.4 ± 55.3 | 61.1 | -95.5 | 0.0 |
| 9 | {"gamma": 0.995} | -8.4 ± 39.1 | 76.1 | -46.6 | 0.0 |
| 10 | {"entropy_coeff": 0.0} | -28.7 ± 47.0 | 67.8 | -70.0 | 0.0 |
| 11 | {"entropy_coeff": 0.05} | -18.3 ± 39.0 | 73.3 | -42.9 | 0.0 |
| 12 | {"steps_per_env": 32, "minibatch_size": 64} | -17.2 ± 47.0 | 72.4 | -43.6 | 0.0 |

Defaults: lr=0.0003, gamma=0.97, hidden=64, grad_clip=0.5, n_envs=8.0, entropy_coeff=0.01, critic_coeff=0.5, gae_lambda=0.95, steps_per_env=128.0, epochs=4.0, minibatch_size=256.0, clip_epsilon=0.2, normalize_advantage=True

### A2C

| Run | Changed vs default | Return (mean±sd) | On-time % | Stress return | Opt-out % |
|---|---|---|---|---|---|
| 1 | default | -48.1 ± 62.7 | 64.5 | -116.3 | 0.0 |
| 2 | {"lr": 0.00025} | -56.3 ± 60.1 | 60.8 | -122.2 | 0.0 |
| 3 | {"lr": 0.002} | -33.4 ± 52.6 | 68.2 | -76.3 | 0.0 |
| 4 | {"steps_per_env": 5} | -39.7 ± 59.7 | 67.8 | -63.2 | 0.0 |
| 5 | {"steps_per_env": 32} | -20.6 ± 39.1 | 75.9 | -42.5 | 0.0 |
| 6 | {"gae_lambda": 0.95} | -37.8 ± 57.2 | 65.6 | -106.3 | 0.0 |
| 7 | {"gamma": 0.9} | -67.9 ± 71.7 | 54.9 | -125.1 | 0.0 |
| 8 | {"gamma": 0.995} | -34.0 ± 59.4 | 66.4 | -85.2 | 0.0 |
| 9 | {"entropy_coeff": 0.0} | -41.6 ± 59.5 | 68.0 | -74.4 | 0.0 |
| 10 | {"entropy_coeff": 0.05} | -42.7 ± 60.3 | 64.5 | -73.8 | 0.0 |
| 11 | {"critic_coeff": 1.0, "normalize_advantage": true} | -58.6 ± 76.1 | 62.3 | -131.3 | 0.0 |
| 12 | {"n_envs": 16, "hidden": 128} | -37.0 ± 59.4 | 69.1 | -114.7 | 0.0 |

Defaults: lr=0.0007, gamma=0.97, hidden=64, grad_clip=0.5, n_envs=8.0, entropy_coeff=0.01, critic_coeff=0.5, gae_lambda=1.0, steps_per_env=8.0, normalize_advantage=False

### Seed robustness (best config, seeds 0-2)

| Algorithm | Run | Return mean over seeds | sd across seeds | On-time % |
|---|---|---|---|---|
| DQN | dqn_run08 | -27.9 | 5.7 | 67.1 |
| REINFORCE | reinforce_run11 | -21.3 | 8.0 | 65.0 |
| PPO | ppo_run06 | -4.5 | 5.7 | 75.4 |
| A2C | a2c_run05 | -27.0 | 6.4 | 67.6 |

### Rule-based references (same 200 evaluation businesses)

- heuristic: train -78.8 (on-time 52.4%), stress -163.2 (on-time 30.1%, opt-out 0.0%)
- random: train -57.2 (on-time 65.9%), stress -106.6 (on-time 45.4%, opt-out 57.0%)
- wait: train -111.7 (on-time 36.6%), stress -180.7 (on-time 20.9%, opt-out 0.0%)