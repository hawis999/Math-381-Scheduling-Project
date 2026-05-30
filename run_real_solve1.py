"""Test harness for real_solve1.py."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from data import load_scenario
from real_solve1 import solve
from solution import Solution, validate2


DEFAULT_DATA_DIR = Path("/Users/haris/Downloads/synthetic_scheduling_data")

SCENARIOS = [
    "01_small_feasible",
    "02_medium_feasible",
    "03_tight_feasible",
    "04_room_bottleneck_optional_drop",
    "05_infeasible_required_course",
    "06_large_stress",
    "07_lp_relaxation_trap",
]


def print_schedule(sol: Solution, data) -> None:
    for prof_id, course_id, ts_id in sorted(sol.selected_triples, key=lambda t: (t[1], t[2], t[0])):
        prof = data.ta_by_id[prof_id]
        course = data.course_by_id[course_id]
        timeslot = data.timeslot_by_id[ts_id]
        print(
            f"  {course.course_code:12s} -> {prof.name:24s} "
            f"{timeslot.day:4s} {timeslot.start_time}-{timeslot.end_time}"
        )


def run_scenario(scenario_dir: Path, show_schedule: bool = False) -> bool:
    name = scenario_dir.name
    data = load_scenario(scenario_dir)

    print(f"\n=== {name} ===")
    print(
        f"  professors: {len(data.tas)}   courses: {len(data.courses)}   "
        f"timeslots: {len(data.timeslots)}   qualified triples: {len(data.qualified_triples)}"
    )

    start = time.time()
    try:
        sol = solve(data)
    except NotImplementedError as e:
        print(f"  not implemented: {e}")
        return False
    except Exception:
        print("  SOLVER RAISED:")
        traceback.print_exc()
        return False
    elapsed = time.time() - start

    if not isinstance(sol, Solution):
        print(f"  ERROR: solve() returned {type(sol).__name__}, expected Solution")
        return False

    print(f"  status:    {sol.status}")
    print(f"  objective: {sol.objective}")
    print(f"  selected:  {len(sol.selected_triples)} (professor, course, timeslot) triples")
    print(f"  wall time: {elapsed:.2f}s")

    ok = True
    vr = validate2(sol, data)
    for error in vr.errors:
        print(f"  FAIL: {error}")
    for warning in vr.warnings:
        print(f"  warn: {warning}")
    if not vr.ok:
        ok = False

    if show_schedule and sol.selected_triples:
        print("  schedule:")
        print_schedule(sol, data)

    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real_solve1.py across scenarios.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--scenario", help="run only this scenario, e.g. 01_small_feasible")
    parser.add_argument("--show-schedule", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    results: dict[str, bool] = {}
    for scenario in scenarios:
        scenario_dir = data_dir / scenario
        if not scenario_dir.exists():
            print(f"skipping {scenario} - not found at {scenario_dir}")
            continue
        results[scenario] = run_scenario(scenario_dir, show_schedule=args.show_schedule)

    print("\n=== SUMMARY ===")
    for scenario, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {scenario}")
    failed = sum(1 for ok in results.values() if not ok)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
