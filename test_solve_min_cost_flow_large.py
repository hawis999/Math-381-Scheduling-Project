"""Large synthetic test for solve_min_cost_flow.py's step-1 model.

This builds the smaller assignment problem in memory:

    professors -> classes

It intentionally avoids timeslots and rooms so you can test the scalable
professor/class assignment subproblem before adding richer scheduling pieces.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

from data import Course, Preference, ScenarioData, TA
from solution import Solution, validate
from solve_min_cost_flow import solve


def weighted_rank(rng: random.Random) -> int | None:
    return rng.choices(
        [1, 2, 3, 4, None],
        weights=[10, 20, 30, 25, 15],
        k=1,
    )[0]


def build_large_step1_instance(
    seed: int,
    professor_count: int,
    course_count: int,
    qualification_prob: float,
    min_load: int,
    max_load: int,
    min_demand: int,
    max_demand: int,
    min_qualified_per_course: int,
) -> ScenarioData:
    rng = random.Random(seed)

    professors = [
        TA(
            ta_id=f"P{i:05d}",
            name=f"Professor {i:05d}",
            max_load=rng.randint(min_load, max_load),
        )
        for i in range(1, professor_count + 1)
    ]
    professor_ids = [prof.ta_id for prof in professors]

    courses: list[Course] = []
    demands: dict[str, int] = {}
    for i in range(1, course_count + 1):
        course_id = f"C{i:05d}"
        demand = rng.randint(min_demand, max_demand)
        demands[course_id] = demand
        courses.append(
            Course(
                course_id=course_id,
                course_code=f"SYN {i:05d}",
                course_title=f"Synthetic Course {i:05d}",
                enrollment=demand,
                is_required=True,
                priority=10,
            )
        )

    total_capacity = sum(prof.max_load for prof in professors)
    total_demand = sum(demands.values())
    if total_capacity < total_demand:
        raise ValueError(
            f"generated total capacity {total_capacity} < total demand {total_demand}; "
            "increase professor count/max load or lower course demand"
        )

    # Plant one feasible assignment first so the random instance is not
    # accidentally impossible because a course has no usable professors.
    remaining_load = {prof.ta_id: prof.max_load for prof in professors}
    professor_slots: list[str] = []
    for prof in professors:
        professor_slots.extend([prof.ta_id] * prof.max_load)
    rng.shuffle(professor_slots)

    qualified_by_course: dict[str, set[str]] = defaultdict(set)
    for course in courses:
        held_back: list[str] = []
        while len(qualified_by_course[course.course_id]) < demands[course.course_id]:
            if not professor_slots:
                raise ValueError("ran out of professor capacity while planting feasibility")
            professor_id = professor_slots.pop()
            if professor_id in qualified_by_course[course.course_id]:
                held_back.append(professor_id)
                continue
            qualified_by_course[course.course_id].add(professor_id)
            remaining_load[professor_id] -= 1
        professor_slots.extend(held_back)

    target_qualified_per_course = max(
        min_qualified_per_course,
        int(round(professor_count * qualification_prob)),
        max_demand,
    )
    target_qualified_per_course = min(target_qualified_per_course, professor_count)

    for course in courses:
        qualified = qualified_by_course[course.course_id]
        while len(qualified) < target_qualified_per_course:
            qualified.add(rng.choice(professor_ids))

    preferences: list[Preference] = []
    qualified_pairs: set[tuple[str, str]] = set()
    pref_by_pair: dict[tuple[str, str], Preference] = {}
    qualified_tas_for: dict[str, list[str]] = {}
    qualified_courses_for: dict[str, list[str]] = defaultdict(list)

    for course in courses:
        course_professors = sorted(qualified_by_course[course.course_id])
        qualified_tas_for[course.course_id] = course_professors
        for professor_id in course_professors:
            pref = Preference(
                ta_id=professor_id,
                course_id=course.course_id,
                qualified=True,
                rank=weighted_rank(rng),
            )
            preferences.append(pref)
            pair = (professor_id, course.course_id)
            qualified_pairs.add(pair)
            pref_by_pair[pair] = pref
            qualified_courses_for[professor_id].append(course.course_id)

    data = ScenarioData(
        name=f"synthetic_step1_flow_seed_{seed}",
        metadata={
            "seed": seed,
            "professors": professor_count,
            "courses": course_count,
            "qualification_probability": qualification_prob,
            "total_capacity": total_capacity,
            "total_demand": total_demand,
            "target_qualified_per_course": target_qualified_per_course,
            "model": "step1_min_cost_max_flow",
        },
        tas=professors,
        courses=courses,
        timeslots=[],
        rooms=[],
        preferences=preferences,
        time_preferences=[],
        room_suitabilities=[],
    )
    data.min_tas = demands
    data.qualified_pairs = qualified_pairs
    data.pref_by_pair = pref_by_pair
    data.qualified_tas_for = qualified_tas_for
    data.qualified_courses_for = dict(qualified_courses_for)
    data.ta_by_id = {prof.ta_id: prof for prof in professors}
    data.course_by_id = {course.course_id: course for course in courses}
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a large synthetic professor-class min-cost flow test."
    )
    parser.add_argument("--seed", type=int, default=381)
    parser.add_argument("--professors", type=int, default=300)
    parser.add_argument("--courses", type=int, default=1000)
    parser.add_argument("--qualification-prob", type=float, default=0.04)
    parser.add_argument("--min-load", type=int, default=4)
    parser.add_argument("--max-load", type=int, default=10)
    parser.add_argument("--min-demand", type=int, default=1)
    parser.add_argument("--max-demand", type=int, default=2)
    parser.add_argument("--min-qualified-per-course", type=int, default=8)
    parser.add_argument("--show-sample", action="store_true")
    return parser.parse_args()


def print_solution_summary(solution: Solution, data: ScenarioData) -> None:
    validation = validate(solution, data)

    print("\n=== MIN-COST FLOW RESULT ===")
    print(f"  status:     {solution.status}")
    print(f"  objective:  {solution.objective}")
    print(f"  selected:   {len(solution.selected_pairs)} professor-class pairs")
    print(f"  runtime:    {solution.solver_time_sec:.4f}s")
    if solution.notes:
        print(f"  notes:      {solution.notes}")

    for error in validation.errors[:20]:
        print(f"  FAIL: {error}")
    if len(validation.errors) > 20:
        print(f"  ... {len(validation.errors) - 20} more validation errors")
    print(f"  valid:      {'yes' if validation.ok else 'no'}")


def main() -> None:
    args = parse_args()

    start = time.perf_counter()
    data = build_large_step1_instance(
        seed=args.seed,
        professor_count=args.professors,
        course_count=args.courses,
        qualification_prob=args.qualification_prob,
        min_load=args.min_load,
        max_load=args.max_load,
        min_demand=args.min_demand,
        max_demand=args.max_demand,
        min_qualified_per_course=args.min_qualified_per_course,
    )
    build_time = time.perf_counter() - start

    print("=== SYNTHETIC STEP-1 INSTANCE ===")
    print(f"  professors:       {len(data.tas)}")
    print(f"  courses:          {len(data.courses)}")
    print(f"  total demand:     {sum(data.min_tas.values())}")
    print(f"  total capacity:   {sum(prof.max_load for prof in data.tas)}")
    print(f"  qualified pairs:  {len(data.qualified_pairs)}")
    print(f"  build time:       {build_time:.4f}s")

    solution = solve(data)
    print_solution_summary(solution, data)

    if args.show_sample and solution.selected_pairs:
        print("\n=== SAMPLE ASSIGNMENTS ===")
        for professor_id, course_id in solution.selected_pairs[:25]:
            print(f"  {professor_id} -> {course_id}")


if __name__ == "__main__":
    main()
