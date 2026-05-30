"""Test harness for step 2: runs solve2.solve() across one or all scenarios."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from data import load_scenario
from solution import Solution, validate2
from solve2 import solve

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


def run_scenario(scenario_dir: Path) -> bool:
    name = scenario_dir.name
    data = load_scenario(scenario_dir)

    print(f"\n=== {name} ===")
    print(f"  TAs: {len(data.tas)}   courses: {len(data.courses)}   "
          f"timeslots: {len(data.timeslots)}   qualified triples: {len(data.qualified_triples)}")

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
    print(f"  selected:  {len(sol.selected_triples)} (TA, course, timeslot) triples")
    print(f"  wall time: {elapsed:.2f}s")

    ok = True
    vr = validate2(sol, data)
    for e in vr.errors:
        print(f"  FAIL: {e}")
    for w in vr.warnings:
        print(f"  warn: {w}")
    if not vr.ok:
        ok = False

    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run step-2 solve() across scenarios.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--scenario", help="run only this scenario (e.g. 01_small_feasible)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    results: dict[str, bool] = {}
    for s in scenarios:
        scenario_dir = data_dir / s
        if not scenario_dir.exists():
            print(f"skipping {s} — not found at {scenario_dir}")
            continue
        results[s] = run_scenario(scenario_dir)

    print("\n=== SUMMARY ===")
    for s, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {s}")
    failed = sum(1 for ok in results.values() if not ok)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
