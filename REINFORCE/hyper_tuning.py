import csv
import itertools
import subprocess
from pathlib import Path

"""
RESULTS:
Baseline 0: gamma=0.995, lr_policy=0.01, lr_baseline=0.001, baseline=0, hidden=64, best_score=497.81
Baseline 20: gamma=0.995, lr_policy=0.0001, lr_baseline=0.001, baseline=20, hidden=128, best_score=503.13
Baseline dynamic: gamma=0.99, lr_policy=0.01, lr_baseline=0.0001, baseline=dynamic, hidden=64, best_score=526.11
"""


SEARCH_SPACE = {
    "gamma": [0.995],
    "lr_policy": [1e-4],
    "lr_baseline": [1e-3],
    "baseline": ["20"],
    "hidden": [128],
}


def run_training(config: dict, n_episodes: int = 5000) -> float:
    """Run the existing train.py script with a configuration."""
    cmd = [
        "python",
        "REINFORCE/train.py",
        "--n_episodes",
        str(n_episodes),
        "--print_every",
        "10000",
        "--gamma",
        str(config["gamma"]),
        "--lr_policy",
        str(config["lr_policy"]),
        "--lr_baseline",
        str(config["lr_baseline"]),
        "--baseline",
        str(config["baseline"]),
        "--hidden",
        str(config["hidden"]),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    best_score = None
    for line in proc.stdout.splitlines():
        if line.startswith("BEST_MODEL_AVG_SCORE"):
            # line format: BEST_MODEL_AVG_SCORE: 1.23 ± 0.45
            try:
                best_score = float(line.split(":", 1)[1].split("±")[0])
            except ValueError:
                best_score = None
    return best_score if best_score is not None else float("nan")


def main() -> None:
    results = []
    keys = list(SEARCH_SPACE.keys())
    for values in itertools.product(*SEARCH_SPACE.values()):
        config = dict(zip(keys, values))
        score = run_training(config)
        results.append({**config, "best_score": score})
        print(f"Tested {config} => score {score}")

    csv_path = Path("tuning_results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["best_score"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()