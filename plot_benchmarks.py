#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from run_benchmarks.py JSON output."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save plot images.",
    )
    parser.add_argument(
        "--use-mean",
        action="store_true",
        help="Use mean runtime instead of min-based summary for derived metrics when available.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_summaries_by_n(summaries: list[dict]) -> dict[int, list[dict]]:
    grouped = {}
    for s in summaries:
        grouped.setdefault(int(s["N"]), []).append(s)

    for n in grouped:
        grouped[n] = sorted(grouped[n], key=lambda x: int(x["P"]))
    return grouped


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_speedup(grouped: dict[int, list[dict]], output_dir: Path, stem: str) -> None:
    plt.figure(figsize=(8, 5))

    for n, rows in grouped.items():
        p_vals = [int(r["P"]) for r in rows]
        speedup_vals = [float(r["SPEEDUP_min_based"]) for r in rows]
        plt.plot(p_vals, speedup_vals, marker="o", label=f"N={n}")

    # Ideal line
    all_p = sorted({int(r["P"]) for rows in grouped.values() for r in rows})
    plt.plot(all_p, all_p, linestyle="--", marker="x", label="Ideal")

    plt.xlabel("Processes")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Processes")
    plt.xticks(all_p)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_speedup.png", dpi=200)
    plt.close()


def plot_efficiency(grouped: dict[int, list[dict]], output_dir: Path, stem: str) -> None:
    plt.figure(figsize=(8, 5))

    for n, rows in grouped.items():
        p_vals = [int(r["P"]) for r in rows]
        eff_vals = [float(r["EFFICIENCY_min_based"]) for r in rows]
        plt.plot(p_vals, eff_vals, marker="o", label=f"N={n}")

    all_p = sorted({int(r["P"]) for rows in grouped.values() for r in rows})

    plt.xlabel("Processes")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Processes")
    plt.xticks(all_p)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_efficiency.png", dpi=200)
    plt.close()


def plot_comm_ratio(grouped: dict[int, list[dict]], output_dir: Path, stem: str) -> None:
    plt.figure(figsize=(8, 5))

    for n, rows in grouped.items():
        p_vals = [int(r["P"]) for r in rows]
        comm_ratio_vals = [float(r["COMM_RATIO_mean"]) for r in rows]
        plt.plot(p_vals, comm_ratio_vals, marker="o", label=f"N={n}")

    all_p = sorted({int(r["P"]) for rows in grouped.values() for r in rows})

    plt.xlabel("Processes")
    plt.ylabel("Communication Ratio")
    plt.title("Communication Ratio vs Processes")
    plt.xticks(all_p)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_comm_ratio.png", dpi=200)
    plt.close()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    ensure_output_dir(output_dir)

    payload = load_json(input_path)
    summaries = payload.get("summaries", [])
    if not summaries:
        raise ValueError("No 'summaries' found in input JSON.")

    grouped = group_summaries_by_n(summaries)
    stem = input_path.stem

    plot_speedup(grouped, output_dir, stem)
    plot_efficiency(grouped, output_dir, stem)
    plot_comm_ratio(grouped, output_dir, stem)

    print("Saved plots:")
    print(f"  {output_dir / f'{stem}_speedup.png'}")
    print(f"  {output_dir / f'{stem}_efficiency.png'}")
    print(f"  {output_dir / f'{stem}_comm_ratio.png'}")


if __name__ == "__main__":
    main()