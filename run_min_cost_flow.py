"""Run solve_min_cost_flow.py on a generated scenario directory."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from data import load_scenario
from solution import Solution, validate
from solve_min_cost_flow import solve


DEFAULT_DATA_DIR = Path("data/synthetic_scheduling_data")
DEFAULT_SCENARIO = "uw_math_autumn_2026_seed_381"


def run_scenario(scenario_dir: Path) -> bool:
    data = load_scenario(scenario_dir)

    print(f"\n=== {scenario_dir.name} ===")
    print(
        f"  professors: {len(data.tas)}   courses: {len(data.courses)}   "
        f"qualified pairs: {len(data.qualified_pairs)}"
    )

    start = time.perf_counter()
    try:
        solution = solve(data)
    except Exception:
        print("  SOLVER RAISED:")
        traceback.print_exc()
        return False
    elapsed = time.perf_counter() - start

    if not isinstance(solution, Solution):
        print(f"  ERROR: solve() returned {type(solution).__name__}, expected Solution")
        return False

    print(f"  status:    {solution.status}")
    print(f"  objective: {solution.objective}")
    print(f"  selected:  {len(solution.selected_pairs)} professor-course pairs")
    print(f"  wall time: {elapsed:.4f}s")
    if solution.notes:
        print(f"  notes:     {solution.notes}")

    validation = validate(solution, data)
    for error in validation.errors:
        print(f"  FAIL: {error}")
    for warning in validation.warnings:
        print(f"  warn: {warning}")

    print(f"  result: {'PASS' if validation.ok else 'FAIL'}")
    return validation.ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run step-1 min-cost flow solve.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    args = parser.parse_args()

    ok = run_scenario(Path(args.data_dir) / args.scenario)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
