import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_results(path):
    with open(path, "r") as file:
        return json.load(file)


def calculate_change(baseline, candidate):
    difference = candidate - baseline

    if baseline == 0:
        ratio = None
    else:
        ratio = candidate / baseline

    return difference, ratio


parser = argparse.ArgumentParser()

parser.add_argument(
    "--baseline",
    required=True,
    help="Path to baseline results.json",
)

parser.add_argument(
    "--candidate",
    required=True,
    help="Path to candidate results.json",
)

args = parser.parse_args()


baseline_path = PROJECT_ROOT / args.baseline
candidate_path = PROJECT_ROOT / args.candidate


baseline = load_results(baseline_path)
candidate = load_results(candidate_path)


baseline_metrics = baseline["metrics"]
candidate_metrics = candidate["metrics"]


print("\nBenchmark comparison")
print("=" * 70)

print(
    f"Baseline  : {baseline['scenario']}"
)
print(
    f"Candidate : {candidate['scenario']}"
)

print("=" * 70)


metric_names = [
    "success_rate",
    "fall_rate",
    "mean_episode_length",
    "linear_velocity_rmse_mps",
    "yaw_velocity_rmse_radps",
]


for metric in metric_names:

    baseline_value = baseline_metrics[metric]
    candidate_value = candidate_metrics[metric]

    difference, ratio = calculate_change(
        baseline_value,
        candidate_value,
    )

    print(f"\n{metric}")

    print(
        f"  Baseline : {baseline_value:.4f}"
    )

    print(
        f"  Candidate: {candidate_value:.4f}"
    )

    print(
        f"  Change   : {difference:+.4f}"
    )

    if ratio is not None:
        print(
            f"  Ratio    : {ratio:.2f}x"
        )
    else:
        print(
            "  Ratio    : N/A "
            "(baseline is zero)"
        )


print("\n" + "=" * 70)