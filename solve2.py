# TODO add constraints and fix weighting


"""
Step 2: Professor-to-class-to-timeslot assignment.

Decision variable:
    x[(i, c, t)] = 1 if professor i teaches course c at timeslot t, else 0

Availability is handled by preprocessing (qualified_triples):
    - professor i qualified for course c
    - professor i available at timeslot t

Objective (you implement) — weighted combination of:
    1. Course preference    COURSE_PREF_WEIGHTS[rank]          (rank 1-4 or None)
    2. Time preference      TIME_WEIGHTS[score]                (score 1-5)
    3. Course priority      COURSE_PRIORITY_WEIGHTS[required]  (True/False)
    4. Room suitability     ROOM_WEIGHTS[score]                (score 1-5, 5=best)

Useful data:
    data.qualified_triples                          -> set of (prof_id, course_id, ts_id)
    data.ta_by_id[prof_id].max_load / .min_load     -> int
    data.pref_by_pair[(prof_id, course_id)].rank    -> int | None
    data.time_pref_score[(prof_id, ts_id)]          -> float 1-5
    data.course_by_id[course_id].is_required        -> bool
    data.best_room_score[course_id]                 -> int 1-5 (best room available)
    data.required_course_ids, data.optional_course_ids

    Precomputed groupings in solve():
        by_ta[prof_id]              -> list of triples
        by_course[course_id]        -> list of triples
        by_ta_time[(prof_id, ts)]   -> list of triples (use for double-booking constraint)

Return Solution(status, objective, selected_triples, solver_time_sec).
Use solution.gurobi_status(model) to translate model.Status.
"""

from __future__ import annotations

from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from data import ScenarioData
from solution import Solution, gurobi_status, preference_weight

# 1. Course preference weights (rank from professor_course_preferences.csv, 1=most preferred)
COURSE_PREF_WEIGHTS: dict[int | None, float] = {
    1: 0.35,
    2: 0.25,
    3: 0.15,
    4: 0.10,
    None: 0.05,  # qualified but no ranked preference
}

# 2. Time preference weights (time_pref_score from professor_time_preferences.csv, 5=best)
TIME_WEIGHTS: dict[int, float] = {
    5: 0.4,  # TODO: best time
    4: 0.3,  # TODO
    3: 0.2,  # TODO
    2: 0.1,  # TODO
    1: 0.05,  # TODO: worst available time
}

# 3. Course priority weights (is_required=True means mandatory)
COURSE_PRIORITY_WEIGHTS: dict[bool, float] = {
    True: 2.5,   # TODO: required — must be taught
    False: 0.25,  # TODO: optional — can be dropped
}

# 4. Room suitability weights (best_room_score for the course, 5=most suitable room available)
ROOM_WEIGHTS: dict[int, float] = {
    5: 1.0,  # TODO: best room
    4: 0.85,  # TODO
    3: 0.7,  # TODO # min required room
    2: 0.25,  # TODO
    1: 0.15,  # TODO: worst suitable room
}


def course_pref_weight(prof_id: str, course_id: str, data) -> float:
    pref = data.pref_by_pair.get((prof_id, course_id))
    return COURSE_PREF_WEIGHTS.get(pref.rank if pref else None, 0.05)

def time_weight(prof_id: str, ts_id: str, data) -> float:
    score = int(data.time_pref_score.get((prof_id, ts_id), 1))
    return TIME_WEIGHTS.get(score, 0.0)

def course_priority_weight(course_id: str, data) -> float:
    course = data.course_by_id[course_id]
    return COURSE_PRIORITY_WEIGHTS[course.is_required]

def room_weight(course_id: str, data) -> float:
    score = data.best_room_score.get(course_id, 1)
    return ROOM_WEIGHTS.get(score, 0.0)


def solve(data: ScenarioData) -> Solution:
    m = gp.Model()
    m.setParam("OutputFlag", 0)

    # One binary variable per qualified (TA, course, timeslot) triple.
    # Availability (course qualification + timeslot availability) is enforced by omission.
    x = {triple: m.addVar(vtype=GRB.BINARY) for triple in data.qualified_triples}

    # Precomputed groupings to make constraint loops clean.
    by_ta: dict[str, list] = defaultdict(list)
    by_course: dict[str, list] = defaultdict(list)
    by_ta_time: dict[tuple[str, str], list] = defaultdict(list)

    for triple in data.qualified_triples:
        ta_id, c_id, ts_id = triple
        by_ta[ta_id].append(triple)
        by_course[c_id].append(triple)
        by_ta_time[(ta_id, ts_id)].append(triple)

    # -----------------------------------------------------------------------
    # ADD YOUR CONSTRAINTS HERE
    # -----------------------------------------------------------------------

    # same constraints as 1

    # min ta's needed per class
    for c in data.courses:
        qualified = data.qualified_tas_for.get(c.course_id, [])
        m.addConstr(
            gp.quicksum(x[(ta_id, c.course_id)] for ta_id in qualified) >= data.min_tas[c.course_id]
        )
    # max load per ta
    for ta in data.tas:
      qualified = data.qualified_courses_for.get(ta.ta_id, [])
      m.addConstr(
         gp.quicksum(x[(ta.ta_id, c)] for c in qualified) <= ta.max_load
      )

    # ---- new constraints ----

    #optional classes taught <= 1 times
    
    # professor not double booked 

    # professors have max classes they can teach 
    
    # all the mandatory classes get taught
    for c_id in data.required_course_ids:
        m.addConstr(
            gp.quicksum(x[triple] for triple in by_course[c_id]) >= 1
        )
    
        

    # -----------------------------------------------------------------------
    # ADD YOUR OBJECTIVE HERE
    # -----------------------------------------------------------------------

    # max 

    m.optimize()

    status = gurobi_status(m)
    selected = [triple for triple, v in x.items() if v.X > 0.5] if m.SolCount > 0 else []
    return Solution(
        status=status,
        objective=m.ObjVal if m.SolCount > 0 else None,
        selected_triples=selected,
        solver_time_sec=m.Runtime,
    )
