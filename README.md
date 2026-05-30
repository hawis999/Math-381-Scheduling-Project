# MATH 381 IP — Step 1: TA to Class assignment

Test harness for the simplest version of the problem: assign TAs to classes
to maximize preference, subject to per-class demand and per-TA load.

## Model (step 1)

Decision variable:

    x[i, c] = 1 if TA i is assigned to class c, else 0

Constraints:

    sum_i x[i, c] >= min_tas[c]               for each course c
    sum_c x[i, c] <= max_load[i]              for each TA i
    x[i, c] = 0 if TA i is not qualified for c

Objective:

    maximize sum_{(i,c)} preference_weight(rank[i, c]) * x[i, c]

Preference weights:

| rank | weight |
|------|--------|
| 1    | 0.35   |
| 2    | 0.25   |
| 3    | 0.15   |
| 4    | 0.10   |
| none | 0.05   |

## Files

| file          | purpose                                                        |
|---------------|----------------------------------------------------------------|
| `data.py`     | Load a scenario into TAs / Courses / Preferences + indexes.    |
| `solution.py` | `Solution` dataclass, hard-constraint validator, pref weights. |
| `solve.py`    | **You implement `solve(data) -> Solution` here.**              |
| `run.py`      | Runs `solve()` on one or all scenarios, validates, reports.    |

## Setup

```bash
pip install gurobipy
```

## Usage

```bash
# one scenario while iterating
python3 run.py --scenario 01_small_feasible

# all 7 scenarios
python3 run.py

# point at a different data location
python3 run.py --data-dir /path/to/synthetic_scheduling_data
```

## Matching simulation

When the full IP has too many variables, you can compare lighter matching
heuristics without Gurobi:

```bash
python3 matching_simulation.py --trials 20 --professors 35 --courses 55 --timeslots 8
```

Algorithms included:

| algorithm             | idea |
|-----------------------|------|
| `greedy`              | take the highest-scoring feasible professor/course/timeslot contract first |
| `required_greedy`     | run score-greedy on required courses first, then optional courses |
| `scarcity_greedy`     | cover required courses with the fewest feasible contracts first |
| `deferred_acceptance` | course-proposing stable matching over professor/timeslot contracts |
| `local_search`        | start from scarcity-greedy and try score-improving replacements |

Useful knobs:

```bash
python3 matching_simulation.py \
  --trials 10 \
  --required-weight 100 \
  --optional-weight 0.5 \
  --qualification-prob 0.18 \
  --availability-prob 0.45
```

The output reports objective, required-course coverage, optional coverage,
constraint violations, and runtime for each algorithm.

## UW-style randomized scenarios

Generate reproducible UW Math Autumn 2026-style scenarios with the same
lecture sections each time, but randomized sensible meeting times:

```bash
python3 generate_uw_dataset.py --seed 381
python3 generate_uw_dataset.py --seed 381 --count 5
```

Run the current professor-course-timeslot solver on one generated scenario:

```bash
python3 run_real_solve1.py \
  --data-dir data/synthetic_scheduling_data \
  --scenario uw_math_autumn_2026_seed_381 \
  --show-schedule
```

Each generated scenario keeps the same course sections and changes only the
synthetic schedule and preference layer by seed.

Generate a large UW-style stress scenario that is intentionally too large for
the size-limited Gurobi IP path:

```bash
python3 generate_uw_dataset.py --seed 381 --stress --scale 25
python3 run_real_solve1.py \
  --data-dir data/synthetic_scheduling_data \
  --scenario uw_math_autumn_2026_stress_seed_381_scale_25
```

Run the CSV-backed matching harness on that same stress scenario:

```bash
python3 run_large_stress_test.py \
  --data-dir data/synthetic_scheduling_data \
  --scenario uw_math_autumn_2026_stress_seed_381_scale_25
```

## What the harness checks

For each scenario it:

1. Calls your `solve(data)`.
2. Validates `selected_pairs` against:
   - every course meets its `min_tas[c]` demand,
   - no TA exceeds `max_load[i]`,
   - no unqualified `(TA, course)` pair selected,
   - reported objective matches recomputed `sum(preference_weight)`.
3. Compares your objective to the bundled hidden feasible solution.

## Tunables

- In `data.py`, `STUDENTS_PER_TA = 1000` makes every course need 1 TA.
  Lower it (e.g. 30) to make large classes require more TAs.

## Skeleton

```python
import gurobipy as gp
from gurobipy import GRB
from data import ScenarioData
from solution import Solution, gurobi_status, preference_weight

def solve(data: ScenarioData) -> Solution:
    m = gp.Model()

    # binary variable for every qualified (TA, course) pair
    x = {pair: m.addVar(vtype=GRB.BINARY) for pair in data.qualified_pairs}

    # min TAs per course
    for c in data.courses:
        m.addConstr(
            gp.quicksum(x[(i, c.course_id)] for i in data.qualified_tas_for.get(c.course_id, []))
            >= data.min_tas[c.course_id]
        )

    # max load per TA
    for ta in data.tas:
        m.addConstr(
            gp.quicksum(x[(ta.ta_id, c)] for c in data.qualified_courses_for.get(ta.ta_id, []))
            <= ta.max_load
        )

    # maximize preference
    m.setObjective(
        gp.quicksum(preference_weight(data.pref_by_pair[pair]) * var
                    for pair, var in x.items()),
        GRB.MAXIMIZE,
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
```
