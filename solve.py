"""
Notes: we prune the dataset ahead of time so that way we don't have to model the availibility of Ta i have class C (A_{i,c})
Pruning beforehand also allows to organize the preference score by weights that way we don't run into the issue of optimal solution being
- 1 class w/ pref 4 versus 3 class w/pref 1 for each of the classes
Thus, we don't need a constraint for the preference score

Step 1: TA-to-class assignment.

Decision variable:
    x[(i, c)] = 1 if TA i is assigned to class c, else 0

NOTE on availability: we only create a variable for (i, c) pairs where TA i is
qualified for course c. This is preprocessing — the availability "constraint"
becomes implicit (the variable just doesn't exist for unqualified pairs).

Constraints:
    sum_i x[(i, c)] >= data.min_tas[c]              for each course c
    sum_c x[(i, c)] <= data.ta_by_id[i].max_load    for each TA i

Objective:
    maximize  sum over (i, c) of  preference_weight(data.pref_by_pair[(i, c)]) * x[(i, c)]

Preference weights (from solution.PREF_WEIGHTS):
    rank 1 -> 0.35
    rank 2 -> 0.25
    rank 3 -> 0.15
    rank 4 -> 0.10
    no rank -> 0.05

Useful inputs (see data.py):
    data.tas, data.courses, data.preferences
    data.min_tas[course_id]                   -> int, min TAs needed
    data.ta_by_id[ta_id].max_load             -> int, max classes for that TA
    data.qualified_pairs                      -> set of (ta_id, course_id) where qualified
    data.qualified_tas_for[course_id]         -> list[str] of TA ids
    data.qualified_courses_for[ta_id]         -> list[str] of course ids
    data.pref_by_pair[(ta_id, course_id)]     -> Preference (has .rank)

Return Solution(status, objective, selected_pairs, solver_time_sec).
Use solution.gurobi_status(model) to translate model.Status.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from data import ScenarioData
from solution import Solution, gurobi_status, preference_weight
import pprint as pp


def solve(data: ScenarioData) -> Solution:
    m = gp.Model()
    m.setParam("OutputFlag", 0)  # silence Gurobi output; set to 1 to see solver log

    # One binary variable per qualified (TA, course) pair.
    # Unqualified pairs have no variable — availability is enforced by omission.
    x = {pair: m.addVar(vtype=GRB.BINARY) for pair in data.qualified_pairs}

    # -----------------------------------------------------------------------
    # ADD YOUR CONSTRAINTS HERE
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # ADD YOUR OBJECTIVE HERE
    # -----------------------------------------------------------------------

    m.setObjective(
        gp.quicksum(preference_weight(data.pref_by_pair[pair]) * x[pair] for pair in x), #maximize preference + (0,1)
        GRB.MAXIMIZE
    )

    m.optimize()

    status = gurobi_status(m)
    selected = [pair for pair, v in x.items() if v.X > 0.5] if m.SolCount > 0 else []
    return Solution(
        status=status,
        objective=m.ObjVal if m.SolCount > 0 else None,
        selected_pairs=selected,
        solver_time_sec=m.Runtime,
    )
