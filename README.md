# Humanoid Simulation Robustness Benchmark

A reproducible robotics simulation and evaluation project for **Unitree H1 humanoid locomotion in NVIDIA Isaac Lab**.

The project evaluates how a frozen locomotion policy behaves when the simulated physical model is deliberately changed. The focus is not on proposing a new PPO algorithm, but on building a **robustness, reproducibility, and regression-testing layer** around an existing humanoid locomotion task.

## Project Scope

**35 multi-seed simulation runs · 3,500 evaluated episodes · 7 physics conditions · 5 evaluation seeds**

The baseline policy was trained using the built-in Isaac Lab H1 locomotion task and RSL-RL PPO. The main contribution of this repository is the independent benchmark infrastructure around that policy.

### Provided by Isaac Lab / RSL-RL

- Unitree H1 robot model
- `Isaac-Velocity-Flat-H1-v0` locomotion environment
- observations, actions, rewards, and task termination logic
- PPO implementation through RSL-RL

### Implemented in this project

- trained and froze a dedicated H1 locomotion policy
- external benchmark runner independent of the standard playback script
- configuration-driven evaluation through YAML
- episode-consistent metric collection
- timeout-survival versus early-termination classification
- velocity-tracking RMSE aligned with the H1 task coordinate frames
- mean base-tilt measurement
- soft joint-limit violation tracking
- deterministic contact-friction perturbations
- deterministic whole-robot mass scaling with inertia recomputation
- automated friction and mass sweeps
- resumable multi-seed benchmark execution
- statistical aggregation across evaluation seeds
- JSON result export and plotting utilities
- regression thresholds for benchmark results
- CPU-only automated tests and GitHub Actions CI

## Baseline Policy

| Parameter | Value |
|---|---:|
| Robot | Unitree H1 |
| Task | `Isaac-Velocity-Flat-H1-v0` |
| Algorithm | PPO — RSL-RL |
| Parallel training environments | 1024 |
| Training iterations | 1500 |
| Total transitions | 36,864,000 |
| Training seed | 42 |
| Final mean reward | 30.96 |
| Final mean episode length | 993.30 |
| Training throughput | ~25,577 steps/s |
| Training time | ~27 min |

Frozen policy:

```text
baseline_checkpoints/baseline_v1.pt
```

Metadata:

```text
baseline_checkpoints/baseline_v1.yaml
```

## Nominal Benchmark

The finalized nominal evaluation uses **100 completed episodes across 64 vectorized environments**.

| Metric | Result |
|---|---:|
| Survival rate | **100.00%** |
| Fall rate | **0.00%** |
| Mean episode length | **1000.00 steps** |
| Linear velocity RMSE | **0.0807 m/s** |
| Yaw velocity RMSE | **0.1395 rad/s** |
| Mean base tilt | **2.99°** |
| Joint-limit violation rate | **0.0979%** |

A successful episode is defined as `survival_to_timeout`: the robot reaches the configured episode time limit without early termination.

This is a locomotion stability metric, not a navigation-goal metric.

## Multi-Seed Robustness Evaluation

The frozen policy was evaluated with:

- **5 seeds:** 42, 123, 456, 789, 2026
- **7 physics scenarios**
- **100 completed episodes per scenario/seed**
- **35 total runs**
- **3,500 evaluated episodes**

Values below are **mean ± sample standard deviation across evaluation seeds**.

### Contact-Friction Robustness

![Friction survival](docs/assets/multiseed_friction_success_rate.png)

| Static / Dynamic Friction | Survival Rate |
|---|---:|
| Nominal 0.8 / 0.6 | **99.8 ± 0.45%** |
| 0.4 / 0.3 | **99.8 ± 0.45%** |
| 0.3 / 0.2 | **81.8 ± 4.76%** |
| 0.2 / 0.15 | **0.0 ± 0.0%** |

The tested friction range shows a clear transition from stable behavior to severe degradation. This is an **observed transition region in the tested grid**, not a universal physical failure threshold.

![Friction yaw RMSE](docs/assets/multiseed_friction_yaw_rmse.png)

| Static / Dynamic Friction | Yaw RMSE |
|---|---:|
| Nominal | **0.141 ± 0.002 rad/s** |
| 0.4 / 0.3 | **0.173 ± 0.002 rad/s** |
| 0.3 / 0.2 | **0.310 ± 0.021 rad/s** |
| 0.2 / 0.15 | **0.684 ± 0.042 rad/s** |

At `0.4 / 0.3`, survival remains near nominal while yaw-tracking error has already increased. This shows why continuous control metrics are useful in addition to binary success/failure metrics.

### Mass-Model Mismatch

Every rigid-body mass in the H1 articulation is scaled while inertia is recomputed and the controller remains unchanged.

![Mass survival](docs/assets/multiseed_mass_success_rate.png)

| Whole-Robot Mass Scale | Survival Rate |
|---|---:|
| 1.0x | **99.8 ± 0.45%** |
| 1.2x | **96.6 ± 1.82%** |
| 1.4x | **85.2 ± 3.77%** |
| 1.6x | **67.2 ± 5.12%** |

Within the tested range, mass scaling produces a more gradual degradation than the tested friction perturbations.

The mass range is intentionally broad and should be interpreted as a **simulation stress test**, not as realistic manufacturing tolerance.

![Mass linear RMSE](docs/assets/multiseed_mass_linear_rmse.png)

| Whole-Robot Mass Scale | Linear Velocity RMSE |
|---|---:|
| 1.0x | **0.086 ± 0.005 m/s** |
| 1.2x | **0.111 ± 0.015 m/s** |
| 1.4x | **0.169 ± 0.014 m/s** |
| 1.6x | **0.248 ± 0.010 m/s** |

![Mass joint-limit violations](docs/assets/multiseed_mass_joint_limit_violations.png)

Larger mass mismatch is also associated with increased soft joint-limit violations. The benchmark measures this association but does not isolate a single causal mechanism.

## Key Findings

1. **Contact friction produced a sharper failure transition than uniform mass scaling within the tested parameter ranges.**
2. **Controller degradation can appear before the robot starts falling.**
3. **Mass mismatch produced progressive degradation across survival, tracking error, and joint-limit behavior.**
4. **The observed trends were reproducible across five evaluation seeds.**

Complete aggregated statistics are stored in:

```text
docs/multiseed_aggregated_results.json
```

## Evaluation Metrics

**Survival rate**  
Fraction of completed benchmark episodes that survive until the configured timeout.

**Fall rate**  
Fraction of completed benchmark episodes that terminate before timeout.

**Linear velocity RMSE**  
XY command tracking RMSE in the gravity-aligned robot yaw frame.

**Yaw velocity RMSE**  
Commanded yaw rate versus the robot's world-frame z angular velocity.

**Mean base tilt**  
Mean angular deviation from upright using projected gravity in the robot base frame.

**Joint-limit violation rate**  
Fraction of joint-time observations outside the configured soft joint-position limits.

## Reproducibility

Development environment:

```text
Isaac Lab:      v2.3.2
RSL-RL:         3.1.2
Python:         3.11
GPU simulation: CUDA
```

Run the nominal benchmark:

```powershell
python benchmark/run_benchmark.py --config configs/baseline_nominal.yaml
```

Run the friction sweep:

```powershell
python scripts/run_friction_sweep.py --config configs/friction_sweep.yaml
```

Run the mass sweep:

```powershell
python scripts/run_mass_sweep.py --config configs/mass_sweep.yaml
```

Run the multi-seed benchmark:

```powershell
python scripts/run_multiseed_benchmark.py --config configs/multiseed_benchmark.yaml
```

Aggregate multi-seed results:

```powershell
python scripts/aggregate_multiseed_results.py
```

## Tests and CI

The simulator-independent benchmark logic is covered by **41 automated tests**.

Run them locally:

```powershell
python -m pytest tests -q
```

Run the nominal regression check:

```powershell
python scripts/check_regression.py
```

Regression thresholds are defined in:

```text
configs/regression_thresholds.yaml
```

GitHub Actions runs the CPU-only validation pipeline on pushes and pull requests.

The CI workflow tests configuration validation, metric logic, statistical aggregation, and regression-check behavior. It intentionally does **not** launch Isaac Sim; GPU simulation remains an explicit benchmark stage.

## Repository Structure

```text
Humanoid-Benchmark/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- baseline_checkpoints/
|   |-- baseline_v1.pt
|   `-- baseline_v1.yaml
|
|-- benchmark/
|   |-- __init__.py
|   |-- aggregation.py
|   |-- core.py
|   |-- regression.py
|   `-- run_benchmark.py
|
|-- configs/
|   |-- baseline_nominal.yaml
|   |-- friction_sweep.yaml
|   |-- mass_sweep.yaml
|   |-- multiseed_benchmark.yaml
|   `-- regression_thresholds.yaml
|
|-- docs/
|   |-- multiseed_aggregated_results.json
|   `-- assets/
|
|-- reports/
|   `-- baseline_v1_nominal/
|       `-- results.json
|
|-- scripts/
|   |-- aggregate_multiseed_results.py
|   |-- check_regression.py
|   |-- compare_results.py
|   |-- plot_friction_sweep.py
|   |-- plot_mass_sweep.py
|   |-- plot_multiseed_results.py
|   |-- run_friction_sweep.py
|   |-- run_mass_sweep.py
|   `-- run_multiseed_benchmark.py
|
|-- tests/
|   |-- test_aggregation.py
|   |-- test_core.py
|   `-- test_regression.py
|
`-- requirements-ci.txt
```

## Tools

- NVIDIA Isaac Lab / Isaac Sim
- Unitree H1
- RSL-RL
- PPO
- PyTorch
- Gymnasium
- Python
- PyTest
- GitHub Actions
- YAML
- Matplotlib
