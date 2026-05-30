"""Large-scenario test harness for the real_solve1 simulated annealing model.

This keeps real_solve1_sa.py as the implementation file and uses it here as
an experiment runner. The default scenario is the larger stress dataset, and
the IP baseline is run in a separate process with a timeout so SA can still
run if the exact solve is too slow.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from data import load_scenario
from real_solve1 import solve as solve_ip
from real_solve1_sa import (
    STARTING_TEMPERATURE_TARGETS,
    AnnealingStats,
    animate_annealing,
    print_annealing_stats,
    print_gap_report,
    print_solution_report,
    print_temperature_comparison,
    run_simulated_annealing,
)
from solution import STATUS_ERROR, Solution


DEFAULT_DATA_DIR = Path("data/synthetic_scheduling_data")
DEFAULT_SCENARIO = "uw_math_autumn_2026_stress_seed_381_scale_25"


@dataclass
class RunSummary:
    seed: int
    starting_temperature: str
    objective: float | None
    valid: bool
    runtime_sec: float | None
    acceptance_rate: float
    final_violations: int


def _ip_worker(scenario_dir: str, queue: mp.Queue) -> None:
    try:
        data = load_scenario(Path(scenario_dir))
        solution = solve_ip(data)
        queue.put(("ok", solution))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def run_ip_with_timeout(scenario_dir: Path, timeout_sec: float) -> Solution:
    """Run the exact IP in a child process so hard instances can time out."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_ip_worker, args=(str(scenario_dir), queue))

    start = time.perf_counter()
    process.start()
    process.join(timeout_sec)
    elapsed = time.perf_counter() - start

    if process.is_alive():
        process.terminate()
        process.join()
        return Solution(
            status=STATUS_ERROR,
            solver_time_sec=elapsed,
            notes=f"IP timed out after {timeout_sec:.1f}s; continuing with SA.",
        )

    if queue.empty():
        return Solution(
            status=STATUS_ERROR,
            solver_time_sec=elapsed,
            notes="IP process exited without returning a solution.",
        )

    status, payload = queue.get()
    if status == "ok":
        solution: Solution = payload
        if solution.solver_time_sec is None:
            solution.solver_time_sec = elapsed
        return solution

    return Solution(
        status=STATUS_ERROR,
        solver_time_sec=elapsed,
        notes=f"IP process raised an exception:\n{payload}",
    )


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test real_solve1_sa.py on a larger scheduling scenario."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("7"),
        help="comma-separated SA seeds, e.g. 1,2,3",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=100,
        help="number of temperature levels for each SA run",
    )
    parser.add_argument(
        "--starting-temperature",
        choices=["low", "medium", "high", "all"],
        default="high",
        help="SA starting-temperature calibration to test",
    )
    parser.add_argument(
        "--skip-ip",
        action="store_true",
        help="do not attempt the exact IP baseline",
    )
    parser.add_argument(
        "--ip-timeout",
        type=float,
        default=30.0,
        help="seconds before giving up on the IP baseline",
    )
    parser.add_argument(
        "--sa-timeout",
        type=float,
        default=None,
        help="optional seconds before stopping each SA run and returning the best state found",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="open the matplotlib animation window for a single SA run",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="run without opening the matplotlib animation window; this is already the default",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=400,
        help="animation frames; history is subsampled like demo_tsp.py",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=25,
        help="milliseconds between animation frames",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="optional path to save the SA animation as .mp4 or .gif",
    )
    return parser.parse_args()


def print_summary_table(rows: list[RunSummary]) -> None:
    if not rows:
        return

    print("\n=== LARGE SA SUMMARY ===")
    print(
        "  seed   temp      objective   valid   runtime_s   "
        "accept_rate   final_viol"
    )
    for row in rows:
        objective = "n/a" if row.objective is None else f"{row.objective:.4f}"
        runtime = "n/a" if row.runtime_sec is None else f"{row.runtime_sec:.2f}"
        print(
            f"  {row.seed:<6d} {row.starting_temperature:<8s} "
            f"{objective:>10s}   {'yes' if row.valid else 'no ':>5s} "
            f"{runtime:>9s}   {100 * row.acceptance_rate:10.2f}% "
            f"{row.final_violations:12d}"
        )


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.data_dir) / args.scenario

    print(f"loading large scenario: {scenario_dir}")
    data = load_scenario(scenario_dir)
    print(
        f"  professors: {len(data.tas)}   courses: {len(data.courses)}   "
        f"required: {len(data.required_course_ids)}   optional: {len(data.optional_course_ids)}"
    )
    print(
        f"  timeslots: {len(data.timeslots)}   "
        f"qualified triples: {len(data.qualified_triples)}"
    )

    ip_solution: Solution | None = None
    if not args.skip_ip:
        print(f"\nrunning IP baseline with {args.ip_timeout:.1f}s timeout...")
        ip_solution = run_ip_with_timeout(scenario_dir, args.ip_timeout)
        print_solution_report("IP BASELINE", ip_solution, data)

    starting_temperatures = (
        list(STARTING_TEMPERATURE_TARGETS)
        if args.starting_temperature == "all"
        else [args.starting_temperature]
    )
    total_runs = len(args.seeds) * len(starting_temperatures)
    wants_visualization = (args.visualize and not args.no_visualize) or args.save is not None
    should_visualize = wants_visualization and total_runs == 1
    if wants_visualization and total_runs != 1:
        print("\nvisualization is skipped when testing multiple seeds or temperatures")

    summaries: list[RunSummary] = []
    stats_for_temperature_table: list[AnnealingStats] = []
    last_history = None

    for seed in args.seeds:
        for starting_temperature in starting_temperatures:
            target_acceptance = STARTING_TEMPERATURE_TARGETS[starting_temperature]
            print(
                f"\nrunning SA on large scenario "
                f"(seed {seed}, starting temperature {starting_temperature}, "
                f"target bad-move acceptance {100 * target_acceptance:.0f}%)..."
            )
            history = [] if should_visualize else None
            sa_solution, stats = run_simulated_annealing(
                data=data,
                seed=seed,
                levels=args.levels,
                starting_temperature=starting_temperature,
                target_bad_move_acceptance=target_acceptance,
                history=history,
                timeout_sec=args.sa_timeout,
            )

            validation = print_solution_report(
                f"SA LARGE ({starting_temperature}, seed {seed})",
                sa_solution,
                data,
            )
            print_annealing_stats(stats)
            print_gap_report(ip_solution, sa_solution)

            summaries.append(
                RunSummary(
                    seed=seed,
                    starting_temperature=starting_temperature,
                    objective=sa_solution.objective,
                    valid=validation.ok,
                    runtime_sec=sa_solution.solver_time_sec,
                    acceptance_rate=stats.acceptance_rate,
                    final_violations=stats.final_constraint_violations,
                )
            )
            stats_for_temperature_table.append(stats)
            last_history = history

    if args.starting_temperature == "all":
        print_temperature_comparison(stats_for_temperature_table)
    print_summary_table(summaries)

    if should_visualize and last_history is not None:
        animate_annealing(last_history, data, args, ip_solution=ip_solution)


if __name__ == "__main__":
    main()
