"""CSV-backed stress harness for matching-style scheduling algorithms.

This loads the same scenario directories as real_solve1.py, converts
qualified triples into contracts, and runs pluggable matching algorithms.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data import ScenarioData, load_scenario
from real_solve1 import assignment_score, solve
from solution import Solution


DEFAULT_DATA_DIR = Path("data/synthetic_scheduling_data")
DEFAULT_SCENARIO = "uw_math_autumn_2026_stress_seed_381_scale_25"


@dataclass(frozen=True)
class Contract:
    professor_id: str
    course_id: str
    timeslot_id: str
    score: float
    rank: int | None
    time_score: int
    is_required: bool


@dataclass
class MatchingInstance:
    data: ScenarioData
    contracts: list[Contract]
    contracts_by_course: dict[str, list[Contract]]


@dataclass
class AlgorithmResult:
    name: str
    selected: list[Contract]
    objective: float
    required_total: int
    required_covered: int
    optional_covered: int
    selected_count: int
    load_violations: int
    double_bookings: int
    runtime_ms: float


ScheduleAlgorithm = Callable[[MatchingInstance], list[Contract]]


def build_instance(data: ScenarioData) -> MatchingInstance:
    contracts: list[Contract] = []
    for professor_id, course_id, timeslot_id in data.qualified_triples:
        pref = data.pref_by_pair.get((professor_id, course_id))
        time_pref = data.time_pref_score.get((professor_id, timeslot_id), 1)
        contracts.append(
            Contract(
                professor_id=professor_id,
                course_id=course_id,
                timeslot_id=timeslot_id,
                score=assignment_score(professor_id, course_id, timeslot_id, data),
                rank=pref.rank if pref else None,
                time_score=int(time_pref),
                is_required=data.course_by_id[course_id].is_required,
            )
        )

    contracts_by_course: dict[str, list[Contract]] = defaultdict(list)
    for contract in contracts:
        contracts_by_course[contract.course_id].append(contract)
    for course_contracts in contracts_by_course.values():
        course_contracts.sort(key=lambda contract: contract.score, reverse=True)

    return MatchingInstance(
        data=data,
        contracts=contracts,
        contracts_by_course=dict(contracts_by_course),
    )


def can_add(
    data: ScenarioData,
    selected_courses: set[str],
    professor_loads: Counter[str],
    professor_times: set[tuple[str, str]],
    contract: Contract,
) -> bool:
    if contract.course_id in selected_courses:
        return False

    if professor_loads[contract.professor_id] >= data.ta_by_id[contract.professor_id].max_load:
        return False

    if (contract.professor_id, contract.timeslot_id) in professor_times:
        return False

    return True


def add_if_feasible(
    data: ScenarioData,
    selected: list[Contract],
    selected_courses: set[str],
    professor_loads: Counter[str],
    professor_times: set[tuple[str, str]],
    contract: Contract,
) -> bool:
    if not can_add(data, selected_courses, professor_loads, professor_times, contract):
        return False
    selected.append(contract)
    selected_courses.add(contract.course_id)
    professor_loads[contract.professor_id] += 1
    professor_times.add((contract.professor_id, contract.timeslot_id))
    return True


def greedy_baseline(instance: MatchingInstance) -> list[Contract]:
    selected: list[Contract] = []
    selected_courses: set[str] = set()
    professor_loads: Counter[str] = Counter()
    professor_times: set[tuple[str, str]] = set()
    for contract in sorted(instance.contracts, key=lambda c: c.score, reverse=True):
        add_if_feasible(
            instance.data,
            selected,
            selected_courses,
            professor_loads,
            professor_times,
            contract,
        )
    return selected


def required_first_greedy(instance: MatchingInstance) -> list[Contract]:
    selected: list[Contract] = []
    selected_courses: set[str] = set()
    professor_loads: Counter[str] = Counter()
    professor_times: set[tuple[str, str]] = set()
    required = [contract for contract in instance.contracts if contract.is_required]
    optional = [contract for contract in instance.contracts if not contract.is_required]
    for contract in sorted(required, key=lambda c: c.score, reverse=True):
        add_if_feasible(
            instance.data,
            selected,
            selected_courses,
            professor_loads,
            professor_times,
            contract,
        )
    for contract in sorted(optional, key=lambda c: c.score, reverse=True):
        add_if_feasible(
            instance.data,
            selected,
            selected_courses,
            professor_loads,
            professor_times,
            contract,
        )
    return selected


ALGORITHMS: dict[str, ScheduleAlgorithm] = {
    "greedy": greedy_baseline,
    "required_greedy": required_first_greedy,
    # Add your matching algorithms here later.
}


def evaluate(name: str, selected: list[Contract], data: ScenarioData, runtime_ms: float) -> AlgorithmResult:
    selected_courses = {contract.course_id for contract in selected}
    required_covered = sum(
        1 for course_id in data.required_course_ids if course_id in selected_courses
    )
    optional_covered = sum(
        1 for course_id in data.optional_course_ids if course_id in selected_courses
    )

    load_counts = Counter(contract.professor_id for contract in selected)
    load_violations = sum(
        max(0, count - data.ta_by_id[professor_id].max_load)
        for professor_id, count in load_counts.items()
    )

    prof_time_counts = Counter(
        (contract.professor_id, contract.timeslot_id) for contract in selected
    )
    double_bookings = sum(max(0, count - 1) for count in prof_time_counts.values())

    return AlgorithmResult(
        name=name,
        selected=selected,
        objective=sum(contract.score for contract in selected),
        required_total=len(data.required_course_ids),
        required_covered=required_covered,
        optional_covered=optional_covered,
        selected_count=len(selected),
        load_violations=load_violations,
        double_bookings=double_bookings,
        runtime_ms=runtime_ms,
    )


def run_algorithm(name: str, algorithm: ScheduleAlgorithm, instance: MatchingInstance) -> AlgorithmResult:
    start = time.perf_counter()
    selected = algorithm(instance)
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate(name, selected, instance.data, runtime_ms)


def format_result(result: AlgorithmResult) -> str:
    required_pct = 100.0 * result.required_covered / max(1, result.required_total)
    violations = result.load_violations + result.double_bookings
    return (
        f"{result.name:18s} "
        f"obj={result.objective:9.2f}  "
        f"required={result.required_covered:5d}/{result.required_total:<5d} "
        f"({required_pct:5.1f}%)  "
        f"optional={result.optional_covered:5d}  "
        f"selected={result.selected_count:5d}  "
        f"viol={violations:3d}  "
        f"time={result.runtime_ms:9.2f}ms"
    )


def try_ip(data: ScenarioData) -> None:
    print("\n=== IP Attempt ===")
    try:
        start = time.perf_counter()
        solution = solve(data)
        elapsed = time.perf_counter() - start
    except Exception:
        print("IP solver raised:")
        traceback.print_exc()
        return

    if not isinstance(solution, Solution):
        print(f"IP returned {type(solution).__name__}, expected Solution")
        return
    print(
        f"status={solution.status} objective={solution.objective} "
        f"selected={len(solution.selected_triples)} time={elapsed:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSV-backed large matching stress tests.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--try-ip", action="store_true", help="also attempt real_solve1.py first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.data_dir) / args.scenario
    data = load_scenario(scenario_dir)
    instance = build_instance(data)

    print(f"=== Scenario: {args.scenario} ===")
    print(f"professors:        {len(data.tas)}")
    print(f"courses:           {len(data.courses)}")
    print(f"required courses:  {len(data.required_course_ids)}")
    print(f"timeslots:         {len(data.timeslots)}")
    print(f"qualified triples: {len(data.qualified_triples)}")
    print(f"contracts:         {len(instance.contracts)}")

    if args.try_ip:
        try_ip(data)

    selected_names = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"unknown algorithm(s): {', '.join(unknown)}")

    print("\n=== Matching Algorithms ===")
    for name in selected_names:
        result = run_algorithm(name, ALGORITHMS[name], instance)
        print(format_result(result))


if __name__ == "__main__":
    main()
