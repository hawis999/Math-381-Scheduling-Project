"""Solution representation, validator, and hidden-solution comparison for step 1."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from data import Preference, ScenarioData

STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ERROR = "ERROR"

# Step-1 preference weights.
PREF_WEIGHTS: dict[int | None, float] = {
    1: 0.35,
    2: 0.25,
    3: 0.15,
    4: 0.10,
    None: 0.05,  # qualified but no ranked preference
}


def preference_weight(pref: Preference) -> float:
    return PREF_WEIGHTS.get(pref.rank, 0.05)


@dataclass
class Solution:
    status: str
    objective: float | None = None
    selected_pairs: list[tuple[str, str]] = field(default_factory=list)         # step 1
    selected_triples: list[tuple[str, str, str]] = field(default_factory=list)  # step 2
    solver_time_sec: float | None = None
    notes: str = ""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_GUROBI_STATUS = {
    2: STATUS_OPTIMAL,
    3: STATUS_INFEASIBLE,
    4: STATUS_INFEASIBLE,
    5: "UNBOUNDED",
    9: STATUS_TIMEOUT,
    11: "INTERRUPTED",
    13: STATUS_FEASIBLE,
}


def gurobi_status(model) -> str:
    return _GUROBI_STATUS.get(model.Status, f"OTHER_{model.Status}")


def validate(solution: Solution, data: ScenarioData) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if solution.status == STATUS_INFEASIBLE:
        return ValidationResult(ok=True)

    if solution.status == STATUS_ERROR:
        return ValidationResult(ok=False, errors=["solver returned ERROR"])

    seen: set[tuple[str, str]] = set()
    for pair in solution.selected_pairs:
        if pair in seen:
            errors.append(f"duplicate pair: {pair}")
        seen.add(pair)
        if pair not in data.qualified_pairs:
            errors.append(f"TA {pair[0]} not qualified for course {pair[1]}")

    course_counts: dict[str, int] = defaultdict(int)
    for ta_id, c_id in solution.selected_pairs:
        course_counts[c_id] += 1
    for c in data.courses:
        needed = data.min_tas[c.course_id]
        got = course_counts[c.course_id]
        if got < needed:
            errors.append(f"course {c.course_id} has {got} TAs, needs >= {needed}")

    ta_counts: dict[str, int] = defaultdict(int)
    for ta_id, c_id in solution.selected_pairs:
        ta_counts[ta_id] += 1
    for ta in data.tas:
        got = ta_counts[ta.ta_id]
        if got > ta.max_load:
            errors.append(
                f"TA {ta.ta_id} assigned {got} courses, exceeds max {ta.max_load}"
            )

    computed = sum(
        preference_weight(data.pref_by_pair[pair])
        for pair in solution.selected_pairs
        if pair in data.pref_by_pair
    )
    if solution.objective is not None and abs(computed - solution.objective) > 1e-4:
        warnings.append(
            f"reported objective {solution.objective} differs from recomputed {computed:.4f}"
        )

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate2(solution: Solution, data: ScenarioData) -> ValidationResult:
    """Validate a step-2 solution (selected_triples)."""
    errors: list[str] = []
    warnings: list[str] = []

    if solution.status == STATUS_INFEASIBLE:
        return ValidationResult(ok=True)
    if solution.status == STATUS_ERROR:
        return ValidationResult(ok=False, errors=["solver returned ERROR"])

    seen: set[tuple[str, str, str]] = set()
    for triple in solution.selected_triples:
        if triple in seen:
            errors.append(f"duplicate triple: {triple}")
        seen.add(triple)
        if triple not in data.qualified_triples:
            errors.append(f"invalid triple {triple}: not in qualified_triples")

    course_counts: dict[str, int] = defaultdict(int)
    ta_counts: dict[str, int] = defaultdict(int)
    ta_time_counts: dict[tuple[str, str], int] = defaultdict(int)

    for ta_id, c_id, ts_id in solution.selected_triples:
        course_counts[c_id] += 1
        ta_counts[ta_id] += 1
        ta_time_counts[(ta_id, ts_id)] += 1

    for c_id in data.required_course_ids:
        if course_counts[c_id] < 1:
            errors.append(f"required course {c_id} not covered")

    for c_id in data.optional_course_ids:
        if course_counts[c_id] > 1:
            errors.append(f"optional course {c_id} covered {course_counts[c_id]} times (max 1)")

    for ta in data.tas:
        if ta_counts[ta.ta_id] > ta.max_load:
            errors.append(f"TA {ta.ta_id} load {ta_counts[ta.ta_id]} exceeds max {ta.max_load}")

    for (ta_id, ts_id), count in ta_time_counts.items():
        if count > 1:
            errors.append(f"TA {ta_id} double-booked at timeslot {ts_id} ({count} classes)")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def hidden_solution_objective(data: ScenarioData, scenario_dir: str | Path) -> float | None:
    """
    Sum of step-1 preference weights for the bundled hidden (TA, course) pairs.
    Returns None if absent or if the hidden assignment doesn't satisfy our min_tas.
    """
    path = Path(scenario_dir) / "hidden_feasible_solution_do_not_optimize_on.csv"
    if not path.exists():
        return None

    total = 0.0
    course_counts: dict[str, int] = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ta_id, c_id = row["professor_id"], row["course_id"]
            course_counts[c_id] += 1
            pref = data.pref_by_pair.get((ta_id, c_id))
            if pref is None:
                return None
            total += preference_weight(pref)

    for c_id, needed in data.min_tas.items():
        if course_counts[c_id] < needed:
            return None  # hidden doesn't satisfy our step-1 demand
    return total
