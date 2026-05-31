"""
Step 2 solver for professor-to-course-to-timeslot assignment.

This file intentionally mirrors the structure of solve2.py, but uses only
triple-indexed variables:

    x[(i, c, t)] = 1 if professor i teaches course c at timeslot t.

Availability and course qualification are handled by preprocessing in
data.qualified_triples.
"""

from __future__ import annotations

from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from data import ScenarioData
from solution import Solution, gurobi_status


# 1. Course preference weights (rank from professor_course_preferences.csv, 1=most preferred)
COURSE_PREF_WEIGHTS: dict[int | None, float] = {
    1: 0.35,
    2: 0.25,
    3: 0.15,
    4: 0.10,
    None: 0.05,
}

# 2. Time preference weights (time_pref_score from professor_time_preferences.csv, 5=best)
TIME_WEIGHTS: dict[int, float] = {
    5: 0.40,
    4: 0.30,
    3: 0.20,
    2: 0.10,
    1: 0.05,
}

# 3. Course priority weights (is_required=True means mandatory)
COURSE_PRIORITY_WEIGHTS: dict[bool, float] = {
    True: 2.50,
    False: 0.25,
}

# 4. Room suitability weights (best_room_score for the course, 5=most suitable room available)
ROOM_WEIGHTS: dict[int, float] = {
    5: 1.00,
    4: 0.85,
    3: 0.70,
    2: 0.25,
    1: 0.15,
}

# 5. Professor preference mode from professors.csv.
# interest_first professors care more about course/topic fit; time_first
# professors care more about the teaching time. The multipliers only affect
# personal preference components, not department priority or room suitability.
PRIORITY_MODE_WEIGHTS: dict[str, tuple[float, float]] = {
    "interest_first": (1.50, 0.75),  # (course preference multiplier, time multiplier)
    "time_first": (0.75, 1.50),
    "balanced": (1.00, 1.00),
}


def course_pref_weight(prof_id: str, course_id: str, data: ScenarioData) -> float:
    pref = data.pref_by_pair.get((prof_id, course_id))
    rank = pref.rank if pref else None
    return COURSE_PREF_WEIGHTS.get(rank, COURSE_PREF_WEIGHTS[None])


def time_weight(prof_id: str, ts_id: str, data: ScenarioData) -> float:
    score = int(data.time_pref_score.get((prof_id, ts_id), 1))
    return TIME_WEIGHTS.get(score, 0.0)


def course_priority_weight(course_id: str, data: ScenarioData) -> float:
    course = data.course_by_id[course_id]
    return COURSE_PRIORITY_WEIGHTS[course.is_required]


def room_weight(course_id: str, data: ScenarioData) -> float:
    score = data.best_room_score.get(course_id, 1)
    return ROOM_WEIGHTS.get(score, 0.0)


def priority_mode_weights(prof_id: str, data: ScenarioData) -> tuple[float, float]:
    prof = data.ta_by_id[prof_id]
    return PRIORITY_MODE_WEIGHTS.get(prof.priority_mode, PRIORITY_MODE_WEIGHTS["balanced"])


def assignment_score(prof_id: str, course_id: str, ts_id: str, data: ScenarioData) -> float:
    """Objective contribution for assigning one professor/course/timeslot triple."""
    course_multiplier, time_multiplier = priority_mode_weights(prof_id, data)
    return (
        course_multiplier * course_pref_weight(prof_id, course_id, data)
        + time_multiplier * time_weight(prof_id, ts_id, data)
        + course_priority_weight(course_id, data)
        + room_weight(course_id, data)
    )


def solve(data: ScenarioData) -> Solution:
    m = gp.Model()
    m.setParam("OutputFlag", 0)

    # One binary variable per qualified (professor, course, timeslot) triple.
    x = {triple: m.addVar(vtype=GRB.BINARY) for triple in data.qualified_triples}

    by_prof: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    by_course: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    by_prof_time: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)

    for triple in data.qualified_triples:
        prof_id, course_id, ts_id = triple
        by_prof[prof_id].append(triple)
        by_course[course_id].append(triple)
        by_prof_time[(prof_id, ts_id)].append(triple)

    # Required courses must be taught exactly once.
    for course_id in data.required_course_ids:
        m.addConstr(
            gp.quicksum(x[triple] for triple in by_course.get(course_id, [])) == 1,
            name=f"required_{course_id}",
        )

    # Optional courses may be taught at most once. (lowkey don't need this constraint)
    for course_id in data.optional_course_ids:
        m.addConstr(
            gp.quicksum(x[triple] for triple in by_course.get(course_id, [])) <= 1,
            name=f"optional_{course_id}",
        )

    # Professors cannot exceed max teaching load.
    for prof in data.tas:
        m.addConstr(
            gp.quicksum(x[triple] for triple in by_prof.get(prof.ta_id, [])) <= prof.max_load,
            name=f"max_load_{prof.ta_id}",
        )

    # Professors cannot teach two courses in the same timeslot.
    for (prof_id, ts_id), triples in by_prof_time.items():
        m.addConstr(
            gp.quicksum(x[triple] for triple in triples) <= 1,
            name=f"no_double_book_{prof_id}_{ts_id}",
        )

    m.setObjective(
        gp.quicksum(
            assignment_score(prof_id, course_id, ts_id, data) * x[(prof_id, course_id, ts_id)]
            for prof_id, course_id, ts_id in data.qualified_triples
        ),
        GRB.MAXIMIZE,
    )

    m.optimize()

    status = gurobi_status(m)
    selected = [triple for triple, var in x.items() if var.X > 0.5] if m.SolCount > 0 else []
    return Solution(
        status=status,
        objective=m.ObjVal if m.SolCount > 0 else None,
        selected_triples=selected,
        solver_time_sec=m.Runtime,
    )
