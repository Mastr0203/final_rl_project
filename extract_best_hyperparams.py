import pandas as pd
from typing import Dict, Any


def extract_best(csv_path: str = "tuning_results.csv") -> Dict[str, Dict[str, Any]]:
    """Return best hyperparameters for each baseline value."""
    df = pd.read_csv(csv_path)
    results = {}
    for baseline in ["0", "20", "dynamic"]:
        subset = df[df["baseline"] == baseline]
        if subset.empty:
            continue
        best_row = subset.loc[subset["best_score"].idxmax()]
        results[baseline] = best_row.to_dict()
    return results


def main(csv_path: str = "tuning_results.csv") -> None:
    results = extract_best(csv_path)
    for base, params in results.items():
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"Baseline {base}: {params_str}")


if __name__ == "__main__":
    main()
