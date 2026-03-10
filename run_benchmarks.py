#!/usr/bin/env python3

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RESULT_PATTERN = re.compile(
    r"RESULT\s+"
    r"N=(?P<N>\d+)\s+"
    r"P=(?P<P>\d+)\s+"
    r"TOTAL=(?P<TOTAL>[0-9eE+\-.]+)\s+"
    r"COMM=(?P<COMM>[0-9eE+\-.]+)\s+"
    r"COMPUTE=(?P<COMPUTE>[0-9eE+\-.]+)\s+"
    r"COMM_RATIO=(?P<COMM_RATIO>[0-9eE+\-.]+)\s+"
    r"CORRECT=(?P<CORRECT>[01])"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated benchmarks for the MPI matrix multiply kernel."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="Matrix sizes N to test, e.g. --sizes 512 1024 2048",
    )
    parser.add_argument(
        "--procs",
        type=int,
        nargs="+",
        required=True,
        help="Process counts to test, e.g. --procs 1 2 4",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of repeated trials per (N, P) configuration.",
    )
    parser.add_argument(
        "--exe",
        type=str,
        default="./matmul_mpi",
        help="Path to benchmark executable.",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="mpirun",
        choices=["mpirun", "srun"],
        help="Launcher to use for parallel runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where JSON output will be saved.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="local",
        help="Tag to include in output filenames, e.g. local or pace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def build_command(launcher: str, exe: str, p: int, n: int) -> list[str]:
    if launcher == "mpirun":
        return [launcher, "-np", str(p), exe, str(n)]
    if launcher == "srun":
        return [launcher, "-n", str(p), exe, str(n)]
    raise ValueError(f"Unsupported launcher: {launcher}")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        match = RESULT_PATTERN.search(line.strip())
        if match:
            data = match.groupdict()
            return {
                "N": int(data["N"]),
                "P": int(data["P"]),
                "TOTAL": float(data["TOTAL"]),
                "COMM": float(data["COMM"]),
                "COMPUTE": float(data["COMPUTE"]),
                "COMM_RATIO": float(data["COMM_RATIO"]),
                "CORRECT": int(data["CORRECT"]),
            }
    raise ValueError("Could not find RESULT line in program output.")


def summarize_trials(records: list[dict]) -> dict:
    totals = [r["TOTAL"] for r in records]
    comms = [r["COMM"] for r in records]
    computes = [r["COMPUTE"] for r in records]
    comm_ratios = [r["COMM_RATIO"] for r in records]
    correct_flags = [r["CORRECT"] for r in records]

    summary = {
        "N": records[0]["N"],
        "P": records[0]["P"],
        "trials": len(records),
        "all_correct": all(flag == 1 for flag in correct_flags),
        "TOTAL_mean": statistics.mean(totals),
        "TOTAL_min": min(totals),
        "TOTAL_max": max(totals),
        "COMM_mean": statistics.mean(comms),
        "COMM_min": min(comms),
        "COMM_max": max(comms),
        "COMPUTE_mean": statistics.mean(computes),
        "COMPUTE_min": min(computes),
        "COMPUTE_max": max(computes),
        "COMM_RATIO_mean": statistics.mean(comm_ratios),
        "COMM_RATIO_min": min(comm_ratios),
        "COMM_RATIO_max": max(comm_ratios),
    }

    if len(records) > 1:
        summary["TOTAL_stdev"] = statistics.stdev(totals)
        summary["COMM_stdev"] = statistics.stdev(comms)
        summary["COMPUTE_stdev"] = statistics.stdev(computes)
        summary["COMM_RATIO_stdev"] = statistics.stdev(comm_ratios)
    else:
        summary["TOTAL_stdev"] = 0.0
        summary["COMM_stdev"] = 0.0
        summary["COMPUTE_stdev"] = 0.0
        summary["COMM_RATIO_stdev"] = 0.0

    return summary


def compute_speedup_and_efficiency(summaries: list[dict]) -> list[dict]:
    by_n = {}
    for s in summaries:
        by_n.setdefault(s["N"], []).append(s)

    enriched = []
    for n, group in by_n.items():
        group_sorted = sorted(group, key=lambda x: x["P"])
        baseline = None
        for item in group_sorted:
            if item["P"] == 1:
                baseline = item["TOTAL_min"]
                break

        if baseline is None:
            raise ValueError(f"No P=1 baseline found for N={n}.")

        for item in group_sorted:
            speedup = baseline / item["TOTAL_min"]
            efficiency = speedup / item["P"]
            new_item = dict(item)
            new_item["SPEEDUP_min_based"] = speedup
            new_item["EFFICIENCY_min_based"] = efficiency
            enriched.append(new_item)

    return sorted(enriched, key=lambda x: (x["N"], x["P"]))


def print_summary_table(summaries: list[dict]) -> None:
    print("\n=== Benchmark Summary (min-based speedup/efficiency) ===")
    header = (
        f"{'N':>6} {'P':>4} {'Correct':>8} "
        f"{'Tmin':>12} {'Tmean':>12} {'CommMean':>12} "
        f"{'CompMean':>12} {'CommRatio':>12} {'Speedup':>10} {'Eff':>10}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s['N']:>6} "
            f"{s['P']:>4} "
            f"{str(s['all_correct']):>8} "
            f"{s['TOTAL_min']:>12.6f} "
            f"{s['TOTAL_mean']:>12.6f} "
            f"{s['COMM_mean']:>12.6f} "
            f"{s['COMPUTE_mean']:>12.6f} "
            f"{s['COMM_RATIO_mean']:>12.6f} "
            f"{s['SPEEDUP_min_based']:>10.3f} "
            f"{s['EFFICIENCY_min_based']:>10.3f}"
        )


def save_json(payload: dict, output_dir: Path, tag: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{tag}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def main():
    args = parse_args()

    exe_path = Path(args.exe)
    if not exe_path.exists():
        print(f"Error: executable not found: {exe_path}", file=sys.stderr)
        sys.exit(1)

    raw_results = []
    grouped_results = {}

    for n in args.sizes:
        for p in args.procs:
            if n % p != 0:
                print(f"Skipping N={n}, P={p} because N % P != 0")
                continue

            trial_records = []
            print(f"\nRunning configuration N={n}, P={p}, trials={args.trials}")

            for trial_idx in range(1, args.trials + 1):
                cmd = build_command(args.launcher, str(exe_path), p, n)
                print(f"  Trial {trial_idx}: {' '.join(cmd)}")

                if args.dry_run:
                    continue

                completed = run_command(cmd)

                if completed.returncode != 0:
                    print("  Command failed.", file=sys.stderr)
                    print("  STDOUT:", completed.stdout, file=sys.stderr)
                    print("  STDERR:", completed.stderr, file=sys.stderr)
                    sys.exit(completed.returncode)

                try:
                    parsed = parse_result(completed.stdout)
                except ValueError as exc:
                    print(f"  Failed to parse output: {exc}", file=sys.stderr)
                    print("  STDOUT:", completed.stdout, file=sys.stderr)
                    print("  STDERR:", completed.stderr, file=sys.stderr)
                    sys.exit(1)

                parsed["trial"] = trial_idx
                parsed["command"] = cmd
                trial_records.append(parsed)
                raw_results.append(parsed)

            if not args.dry_run:
                grouped_results[(n, p)] = trial_records

    if args.dry_run:
        print("\nDry run complete.")
        return

    summaries = []
    for _, records in grouped_results.items():
        summaries.append(summarize_trials(records))

    summaries = compute_speedup_and_efficiency(summaries)
    print_summary_table(summaries)

    payload = {
        "metadata": {
            "launcher": args.launcher,
            "executable": str(exe_path),
            "sizes": args.sizes,
            "procs": args.procs,
            "trials": args.trials,
            "tag": args.tag,
            "generated_at": datetime.now().isoformat(),
        },
        "raw_results": raw_results,
        "summaries": summaries,
    }

    output_path = save_json(payload, Path(args.output_dir), args.tag)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()