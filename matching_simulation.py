"""Compare matching-style heuristics for professor/course/timeslot scheduling.

This is intentionally independent of Gurobi. It gives you a fast way to test
what happens when the full IP has too many parameters and you want lighter
algorithms that still respect the main scheduling constraints.

Example:
    python3 matching_simulation.py --trials 20 --professors 35 --courses 55 --timeslots 8
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Callable


COURSE_PREF_WEIGHTS: dict[int | None, float] = {
    1: 3.5,
    2: 2.5,
    3: 1.5,
    4: 1.0,
    None: 0.5,
}

TIME_WEIGHTS: dict[int, float] = {
    5: 4.0,
    4: 3.0,
    3: 2.0,
    2: 1.0,
    1: 0.5,
}

ROOM_WEIGHTS: dict[int, float] = {
    5: 1.0,
    4: 0.85,
    3: 0.70,
    2: 0.25,
    1: 0.15,
}


@dataclass(frozen=True)
class Professor:
    prof_id: str
    max_load: int


@dataclass(frozen=True)
class Course:
    course_id: str
    is_required: bool
    room_score: int


@dataclass(frozen=True)
class Contract:
    prof_id: str
    course_id: str
    timeslot_id: str
    rank: int | None
    time_score: int
    room_score: int
    score: float


@dataclass
class Instance:
    professors: list[Professor]
    courses: list[Course]
    timeslots: list[str]
    contracts: list[Contract]
    professor_by_id: dict[str, Professor]
    course_by_id: dict[str, Course]
    contracts_by_course: dict[str, list[Contract]]


@dataclass
class AlgorithmResult:
    name: str
    schedule: dict[str, Contract]
    runtime_ms: float
    objective: float
    required_total: int
    required_covered: int
    optional_covered: int
    selected_count: int
    load_violations: int
    double_bookings: int


Schedule = dict[str, Contract]
Algorithm = Callable[[Instance, random.Random], Schedule]


def weighted_choice(items: list[tuple[int | None, float]], rng: random.Random) -> int | None:
    total = sum(weight for _, weight in items)
    mark = rng.random() * total
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if mark <= cumulative:
            return value
    return items[-1][0]


def contract_score(
    course: Course,
    rank: int | None,
    time_score: int,
    required_weight: float,
    optional_weight: float,
    noise: float,
) -> float:
    course_priority = required_weight if course.is_required else optional_weight
    return (
        COURSE_PREF_WEIGHTS[rank]
        + TIME_WEIGHTS[time_score]
        + ROOM_WEIGHTS[course.room_score]
        + course_priority
        + noise
    )


def generate_instance(
    *,
    rng: random.Random,
    num_professors: int,
    num_courses: int,
    num_timeslots: int,
    required_fraction: float,
    qualification_prob: float,
    availability_prob: float,
    max_load: int,
    required_weight: float,
    optional_weight: float,
    ensure_required_has_contract: bool,
) -> Instance:
    professors = [
        Professor(prof_id=f"P{i:03d}", max_load=rng.randint(max(1, max_load - 1), max_load + 1))
        for i in range(num_professors)
    ]
    timeslots = [f"T{i:02d}" for i in range(num_timeslots)]

    required_count = max(1, round(num_courses * required_fraction))
    required_ids = set(rng.sample(range(num_courses), required_count))
    courses = [
        Course(
            course_id=f"C{i:03d}",
            is_required=i in required_ids,
            room_score=rng.randint(1, 5),
        )
        for i in range(num_courses)
    ]

    available_times: dict[str, list[tuple[str, int]]] = {}
    for professor in professors:
        slots: list[tuple[str, int]] = []
        for timeslot in timeslots:
            if rng.random() < availability_prob:
                slots.append((timeslot, rng.randint(1, 5)))
        if not slots:
            timeslot = rng.choice(timeslots)
            slots.append((timeslot, rng.randint(1, 5)))
        available_times[professor.prof_id] = slots

    contracts: list[Contract] = []
    rank_distribution: list[tuple[int | None, float]] = [
        (1, 0.16),
        (2, 0.20),
        (3, 0.22),
        (4, 0.18),
        (None, 0.24),
    ]

    def add_contract(professor: Professor, course: Course, timeslot_id: str, time_score: int) -> None:
        rank = weighted_choice(rank_distribution, rng)
        score = contract_score(
            course=course,
            rank=rank,
            time_score=time_score,
            required_weight=required_weight,
            optional_weight=optional_weight,
            noise=rng.random() * 0.01,
        )
        contracts.append(
            Contract(
                prof_id=professor.prof_id,
                course_id=course.course_id,
                timeslot_id=timeslot_id,
                rank=rank,
                time_score=time_score,
                room_score=course.room_score,
                score=score,
            )
        )

    for professor in professors:
        for course in courses:
            if rng.random() >= qualification_prob:
                continue
            for timeslot_id, time_score in available_times[professor.prof_id]:
                add_contract(professor, course, timeslot_id, time_score)

    if ensure_required_has_contract:
        covered_courses = {contract.course_id for contract in contracts}
        for course in courses:
            if not course.is_required or course.course_id in covered_courses:
                continue
            professor = rng.choice(professors)
            timeslot_id, time_score = rng.choice(available_times[professor.prof_id])
            add_contract(professor, course, timeslot_id, time_score)

    professor_by_id = {professor.prof_id: professor for professor in professors}
    course_by_id = {course.course_id: course for course in courses}
    contracts_by_course: dict[str, list[Contract]] = defaultdict(list)
    for contract in contracts:
        contracts_by_course[contract.course_id].append(contract)
    for course_contracts in contracts_by_course.values():
        course_contracts.sort(key=lambda contract: contract.score, reverse=True)

    return Instance(
        professors=professors,
        courses=courses,
        timeslots=timeslots,
        contracts=contracts,
        professor_by_id=professor_by_id,
        course_by_id=course_by_id,
        contracts_by_course=dict(contracts_by_course),
    )


def objective(schedule: Schedule) -> float:
    return sum(contract.score for contract in schedule.values())


def used_loads(schedule: Schedule) -> Counter[str]:
    return Counter(contract.prof_id for contract in schedule.values())


def used_prof_times(schedule: Schedule) -> set[tuple[str, str]]:
    return {(contract.prof_id, contract.timeslot_id) for contract in schedule.values()}


def can_add(instance: Instance, schedule: Schedule, contract: Contract) -> bool:
    if contract.course_id in schedule:
        return False

    loads = used_loads(schedule)
    if loads[contract.prof_id] >= instance.professor_by_id[contract.prof_id].max_load:
        return False

    if (contract.prof_id, contract.timeslot_id) in used_prof_times(schedule):
        return False

    return True


def can_replace(instance: Instance, schedule: Schedule, course_id: str, contract: Contract) -> bool:
    trial = dict(schedule)
    trial.pop(course_id, None)
    return can_add(instance, trial, contract)


def add_if_feasible(instance: Instance, schedule: Schedule, contract: Contract) -> bool:
    if can_add(instance, schedule, contract):
        schedule[contract.course_id] = contract
        return True
    return False


def greedy_score(instance: Instance, rng: random.Random) -> Schedule:
    schedule: Schedule = {}
    contracts = sorted(instance.contracts, key=lambda contract: contract.score, reverse=True)
    for contract in contracts:
        add_if_feasible(instance, schedule, contract)
    return schedule


def required_first_greedy(instance: Instance, rng: random.Random) -> Schedule:
    schedule: Schedule = {}
    required_contracts = [
        contract
        for contract in instance.contracts
        if instance.course_by_id[contract.course_id].is_required
    ]
    optional_contracts = [
        contract
        for contract in instance.contracts
        if not instance.course_by_id[contract.course_id].is_required
    ]

    for contract in sorted(required_contracts, key=lambda c: c.score, reverse=True):
        add_if_feasible(instance, schedule, contract)
    for contract in sorted(optional_contracts, key=lambda c: c.score, reverse=True):
        add_if_feasible(instance, schedule, contract)
    return schedule


def scarcity_first_greedy(instance: Instance, rng: random.Random) -> Schedule:
    schedule: Schedule = {}
    required_courses = [
        course
        for course in instance.courses
        if course.is_required
    ]
    required_courses.sort(
        key=lambda course: (
            len(instance.contracts_by_course.get(course.course_id, [])),
            -max(
                (contract.score for contract in instance.contracts_by_course.get(course.course_id, [])),
                default=0.0,
            ),
        )
    )

    for course in required_courses:
        for contract in instance.contracts_by_course.get(course.course_id, []):
            if add_if_feasible(instance, schedule, contract):
                break

    optional_contracts = [
        contract
        for contract in instance.contracts
        if not instance.course_by_id[contract.course_id].is_required
    ]
    for contract in sorted(optional_contracts, key=lambda c: c.score, reverse=True):
        add_if_feasible(instance, schedule, contract)

    return schedule


def professor_utility(instance: Instance, contract: Contract) -> float:
    course = instance.course_by_id[contract.course_id]
    required_bonus = 3.0 if course.is_required else 0.0
    return (
        COURSE_PREF_WEIGHTS[contract.rank]
        + TIME_WEIGHTS[contract.time_score]
        + ROOM_WEIGHTS[contract.room_score] * 0.5
        + required_bonus
    )


def choose_professor_contracts(
    instance: Instance,
    prof_id: str,
    contracts: list[Contract],
) -> list[Contract]:
    chosen: list[Contract] = []
    used_timeslots: set[str] = set()
    max_load = instance.professor_by_id[prof_id].max_load

    sorted_contracts = sorted(
        contracts,
        key=lambda contract: professor_utility(instance, contract),
        reverse=True,
    )
    seen_courses: set[str] = set()
    for contract in sorted_contracts:
        if contract.course_id in seen_courses:
            continue
        if contract.timeslot_id in used_timeslots:
            continue
        if len(chosen) >= max_load:
            break
        chosen.append(contract)
        used_timeslots.add(contract.timeslot_id)
        seen_courses.add(contract.course_id)
    return chosen


def deferred_acceptance(instance: Instance, rng: random.Random) -> Schedule:
    course_order = sorted(
        instance.courses,
        key=lambda course: (
            not course.is_required,
            len(instance.contracts_by_course.get(course.course_id, [])),
        ),
    )
    next_proposal = {course.course_id: 0 for course in course_order}
    queue = deque(course.course_id for course in course_order)
    held_by_prof: dict[str, list[Contract]] = defaultdict(list)
    matched_course: dict[str, Contract] = {}

    while queue:
        course_id = queue.popleft()
        if course_id in matched_course:
            continue

        course_contracts = instance.contracts_by_course.get(course_id, [])
        idx = next_proposal[course_id]
        if idx >= len(course_contracts):
            continue

        proposal = course_contracts[idx]
        next_proposal[course_id] = idx + 1
        prof_id = proposal.prof_id

        candidates = held_by_prof[prof_id] + [proposal]
        chosen = choose_professor_contracts(instance, prof_id, candidates)
        chosen_ids = {(c.course_id, c.timeslot_id) for c in chosen}

        rejected = [
            contract
            for contract in candidates
            if (contract.course_id, contract.timeslot_id) not in chosen_ids
        ]

        held_by_prof[prof_id] = chosen
        for contract in chosen:
            matched_course[contract.course_id] = contract

        for contract in rejected:
            matched_course.pop(contract.course_id, None)
            if next_proposal[contract.course_id] < len(
                instance.contracts_by_course.get(contract.course_id, [])
            ):
                queue.append(contract.course_id)

    return matched_course


def local_search(instance: Instance, rng: random.Random) -> Schedule:
    schedule = scarcity_first_greedy(instance, rng)
    contracts = sorted(instance.contracts, key=lambda contract: contract.score, reverse=True)

    for _ in range(20):
        changed = False
        for contract in contracts:
            current = schedule.get(contract.course_id)
            if current is not None:
                if contract.score <= current.score + 1e-9:
                    continue
                if can_replace(instance, schedule, contract.course_id, contract):
                    schedule[contract.course_id] = contract
                    changed = True
                    continue

            if current is None and add_if_feasible(instance, schedule, contract):
                changed = True
                continue

            if current is not None:
                continue

            blockers = blocking_courses(instance, schedule, contract)
            for size in (1, 2):
                for removed_courses in combinations(blockers, size):
                    if not can_drop_courses(instance, schedule, contract, list(removed_courses)):
                        continue
                    trial = dict(schedule)
                    removed_score = 0.0
                    for course_id in removed_courses:
                        removed_score += trial.pop(course_id).score
                    if can_add(instance, trial, contract) and contract.score > removed_score + 1e-9:
                        trial[contract.course_id] = contract
                        schedule = trial
                        changed = True
                        break
                if changed:
                    break
            if changed:
                continue

        if not changed:
            break

    return schedule


def blocking_courses(instance: Instance, schedule: Schedule, contract: Contract) -> list[str]:
    blockers: set[str] = set()
    loads = used_loads(schedule)
    if loads[contract.prof_id] >= instance.professor_by_id[contract.prof_id].max_load:
        for course_id, selected in schedule.items():
            if selected.prof_id == contract.prof_id:
                blockers.add(course_id)

    for course_id, selected in schedule.items():
        if selected.prof_id == contract.prof_id and selected.timeslot_id == contract.timeslot_id:
            blockers.add(course_id)

    return sorted(blockers, key=lambda course_id: schedule[course_id].score)


def can_drop_courses(
    instance: Instance,
    schedule: Schedule,
    new_contract: Contract,
    removed_courses: list[str],
) -> bool:
    new_course = instance.course_by_id[new_contract.course_id]
    required_removed = sum(
        1
        for course_id in removed_courses
        if instance.course_by_id[course_id].is_required
    )
    required_added = 1 if new_course.is_required and new_contract.course_id not in schedule else 0
    return required_removed <= required_added


ALGORITHMS: dict[str, Algorithm] = {
    "greedy": greedy_score,
    "required_greedy": required_first_greedy,
    "scarcity_greedy": scarcity_first_greedy,
    "deferred_acceptance": deferred_acceptance,
    "local_search": local_search,
}


def evaluate(name: str, schedule: Schedule, instance: Instance, runtime_ms: float) -> AlgorithmResult:
    required_courses = [course for course in instance.courses if course.is_required]
    optional_courses = [course for course in instance.courses if not course.is_required]
    required_covered = sum(1 for course in required_courses if course.course_id in schedule)
    optional_covered = sum(1 for course in optional_courses if course.course_id in schedule)

    load_counts = used_loads(schedule)
    load_violations = sum(
        max(0, count - instance.professor_by_id[prof_id].max_load)
        for prof_id, count in load_counts.items()
    )

    prof_time_counts = Counter(
        (contract.prof_id, contract.timeslot_id)
        for contract in schedule.values()
    )
    double_bookings = sum(max(0, count - 1) for count in prof_time_counts.values())

    return AlgorithmResult(
        name=name,
        schedule=schedule,
        runtime_ms=runtime_ms,
        objective=objective(schedule),
        required_total=len(required_courses),
        required_covered=required_covered,
        optional_covered=optional_covered,
        selected_count=len(schedule),
        load_violations=load_violations,
        double_bookings=double_bookings,
    )


def run_algorithm(name: str, algorithm: Algorithm, instance: Instance, rng: random.Random) -> AlgorithmResult:
    start = time.perf_counter()
    schedule = algorithm(instance, rng)
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate(name, schedule, instance, runtime_ms)


def format_result(result: AlgorithmResult) -> str:
    required_pct = 100.0 * result.required_covered / max(1, result.required_total)
    return (
        f"{result.name:20s} "
        f"obj={result.objective:8.2f}  "
        f"required={result.required_covered:3d}/{result.required_total:<3d} "
        f"({required_pct:5.1f}%)  "
        f"optional={result.optional_covered:3d}  "
        f"selected={result.selected_count:3d}  "
        f"viol={result.load_violations + result.double_bookings:2d}  "
        f"time={result.runtime_ms:7.2f}ms"
    )


def summarize(results_by_algorithm: dict[str, list[AlgorithmResult]]) -> None:
    print("\n=== Aggregate Results ===")
    print(
        f"{'algorithm':20s} {'avg obj':>10s} {'req cover':>10s} "
        f"{'avg opt':>8s} {'avg selected':>12s} {'avg ms':>9s}"
    )
    for name, results in results_by_algorithm.items():
        avg_obj = statistics.mean(result.objective for result in results)
        avg_req_cover = statistics.mean(
            result.required_covered / max(1, result.required_total)
            for result in results
        )
        avg_optional = statistics.mean(result.optional_covered for result in results)
        avg_selected = statistics.mean(result.selected_count for result in results)
        avg_ms = statistics.mean(result.runtime_ms for result in results)
        print(
            f"{name:20s} {avg_obj:10.2f} {avg_req_cover:9.1%} "
            f"{avg_optional:8.2f} {avg_selected:12.2f} {avg_ms:9.2f}"
        )


def print_instance_summary(instance: Instance) -> None:
    required_total = sum(1 for course in instance.courses if course.is_required)
    total_capacity = sum(professor.max_load for professor in instance.professors)
    contract_counts = [len(instance.contracts_by_course.get(course.course_id, [])) for course in instance.courses]
    print("=== Instance ===")
    print(f"professors:       {len(instance.professors)}")
    print(f"courses:          {len(instance.courses)} ({required_total} required)")
    print(f"timeslots:        {len(instance.timeslots)}")
    print(f"total capacity:   {total_capacity}")
    print(f"contracts:        {len(instance.contracts)}")
    print(f"contracts/course: min={min(contract_counts)} median={statistics.median(contract_counts)} max={max(contract_counts)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate matching algorithms for professor/course/timeslot scheduling."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=381)
    parser.add_argument("--professors", type=int, default=30)
    parser.add_argument("--courses", type=int, default=45)
    parser.add_argument("--timeslots", type=int, default=8)
    parser.add_argument("--required-fraction", type=float, default=0.55)
    parser.add_argument("--qualification-prob", type=float, default=0.22)
    parser.add_argument("--availability-prob", type=float, default=0.55)
    parser.add_argument("--max-load", type=int, default=3)
    parser.add_argument("--required-weight", type=float, default=10.0)
    parser.add_argument("--optional-weight", type=float, default=0.5)
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHMS),
        help=f"comma-separated subset of: {', '.join(ALGORITHMS)}",
    )
    parser.add_argument(
        "--allow-required-without-contract",
        action="store_true",
        help="leave randomly infeasible required courses in the instance instead of adding one feasible contract",
    )
    parser.add_argument(
        "--show-schedule",
        action="store_true",
        help="print selected contracts for the final trial",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_algorithm_names = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    unknown = [name for name in selected_algorithm_names if name not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"unknown algorithm(s): {', '.join(unknown)}")

    results_by_algorithm: dict[str, list[AlgorithmResult]] = {
        name: []
        for name in selected_algorithm_names
    }
    last_instance: Instance | None = None
    last_results: list[AlgorithmResult] = []

    for trial in range(args.trials):
        rng = random.Random(args.seed + trial)
        instance = generate_instance(
            rng=rng,
            num_professors=args.professors,
            num_courses=args.courses,
            num_timeslots=args.timeslots,
            required_fraction=args.required_fraction,
            qualification_prob=args.qualification_prob,
            availability_prob=args.availability_prob,
            max_load=args.max_load,
            required_weight=args.required_weight,
            optional_weight=args.optional_weight,
            ensure_required_has_contract=not args.allow_required_without_contract,
        )
        last_instance = instance
        last_results = []

        if trial == 0:
            print_instance_summary(instance)

        print(f"=== Trial {trial + 1}/{args.trials} ===")
        for name in selected_algorithm_names:
            result = run_algorithm(name, ALGORITHMS[name], instance, random.Random(args.seed + trial))
            results_by_algorithm[name].append(result)
            last_results.append(result)
            print(format_result(result))
        print()

    summarize(results_by_algorithm)

    if args.show_schedule and last_instance is not None:
        print("\n=== Final Trial Schedules ===")
        for result in last_results:
            print(f"\n{result.name}")
            for contract in sorted(result.schedule.values(), key=lambda c: (c.course_id, c.prof_id, c.timeslot_id)):
                required = "required" if last_instance.course_by_id[contract.course_id].is_required else "optional"
                print(
                    f"  {contract.course_id:5s} {required:8s} -> "
                    f"{contract.prof_id:5s} at {contract.timeslot_id:4s} "
                    f"score={contract.score:5.2f} rank={contract.rank} time={contract.time_score}"
                )


if __name__ == "__main__":
    main()
