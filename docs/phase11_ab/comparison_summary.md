# Phase 11 — Power-Regularized Policy A/B Evaluation

## Experiment

A second Unitree H1 locomotion policy was trained using the same Isaac Lab task and RSL-RL PPO setup as the baseline, with one additional reward term:

`mechanical_power_l1 = sum_j |tau_j * qdot_j|`

Reward weight:

`-5.0e-4`

This term regularizes simulated mechanical actuator power. It is not a model of electrical or battery energy consumption.

The baseline and power-regularized policies were evaluated using the same independent benchmark protocol:

- 5 seeds: 42, 123, 456, 789, 2026
- 100 episodes per seed
- 64 parallel environments
- 4 matched simulation conditions
- 40 total runs
- 4,000 total episodes
- statistics reported as mean ± sample standard deviation across seeds

## Results

| Scenario | Policy | Success (%) | Linear RMSE (m/s) | Yaw RMSE (rad/s) | Tilt (deg) | Joint-limit violation (%) | Mechanical power (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| Nominal | Baseline | 99.80 ± 0.45 | 0.0862 ± 0.0048 | 0.1410 ± 0.0015 | 3.03 ± 0.05 | 0.104 ± 0.005 | 242.29 ± 3.20 |
| Nominal | Power-reg. | 99.80 ± 0.45 | 0.0897 ± 0.0024 | 0.1368 ± 0.0008 | 3.80 ± 0.05 | 0.561 ± 0.026 | 135.83 ± 1.35 |
| Friction 0.3/0.2 | Baseline | 81.80 ± 4.76 | 0.0984 ± 0.0046 | 0.3101 ± 0.0209 | 4.44 ± 0.12 | 0.413 ± 0.050 | 332.99 ± 9.68 |
| Friction 0.3/0.2 | Power-reg. | 46.00 ± 8.77 | 0.1180 ± 0.0098 | 0.3725 ± 0.0251 | 6.52 ± 0.30 | 1.160 ± 0.136 | 200.68 ± 13.00 |
| Mass 1.4x | Baseline | 85.20 ± 3.77 | 0.1691 ± 0.0142 | 0.2079 ± 0.0111 | 3.38 ± 0.12 | 1.483 ± 0.063 | 286.33 ± 11.61 |
| Mass 1.4x | Power-reg. | 70.80 ± 5.50 | 0.1978 ± 0.0079 | 0.2848 ± 0.0199 | 6.44 ± 0.07 | 6.775 ± 0.100 | 134.58 ± 2.40 |
| Actuator effort 0.4x | Baseline | 86.80 ± 6.26 | 0.1401 ± 0.0123 | 0.1813 ± 0.0113 | 3.36 ± 0.18 | 0.548 ± 0.059 | 216.84 ± 5.66 |
| Actuator effort 0.4x | Power-reg. | 94.60 ± 1.52 | 0.1199 ± 0.0025 | 0.1858 ± 0.0069 | 5.23 ± 0.09 | 2.433 ± 0.053 | 140.53 ± 3.37 |

## Mechanical-power reduction

- Nominal: 242.29 W -> 135.83 W (**43.9% lower**)
- Friction 0.3/0.2: 332.99 W -> 200.68 W (**39.7% lower**)
- Mass 1.4x: 286.33 W -> 134.58 W (**53.0% lower**)
- Actuator effort 0.4x: 216.84 W -> 140.53 W (**35.2% lower**)

## Interpretation

Mechanical-power regularization produced substantially lower-power locomotion across every tested condition.

Under nominal dynamics, survival remained unchanged at 99.8% and velocity tracking remained similar, while mean simulated mechanical actuator power decreased by approximately 44%.

The effect on robustness was condition-dependent:

- contact-friction robustness decreased substantially,
- whole-robot mass-mismatch robustness decreased,
- reduced actuator-effort robustness improved.

The lower-power policy also showed higher base tilt and higher joint-limit violation rates, indicating a measurable power-versus-motion-quality/robustness trade-off.

These results should therefore not be interpreted as a universal improvement in locomotion performance. Instead, they demonstrate how reward shaping can shift a learned controller along competing objectives and why independent robustness evaluation is necessary after retraining.

## Scope

Mechanical power is computed from simulated joint torque and velocity as the sum of absolute joint mechanical power.

It should not be interpreted as electrical power, battery consumption, or real-hardware energy efficiency.
