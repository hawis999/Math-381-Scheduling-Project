"""
Learning scaffold: simulated annealing for the real UW solve-1 problem.

This file wires together the boring-but-useful parts for you:
  * loading the UW-style scenario,
  * running the exact Gurobi IP from real_solve1.py,
  * running a generic simulated annealing loop,
  * validating the final schedule,
  * comparing your SA objective against the IP objective.

Your job is to fill in the modeling decisions marked TODO below. Those are
the parts where the actual simulated annealing thinking happens.
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, FrozenSet

from data import ScenarioData, load_scenario
from solution import (
    STATUS_ERROR,
    STATUS_FEASIBLE,
    Solution,
    ValidationResult,
    validate2,
)

DEFAULT_DATA_DIR = Path("data/synthetic_scheduling_data")
DEFAULT_SCENARIO = "uw_math_autumn_2026_seed_381"
STARTING_TEMPERATURE_TARGETS = {
    "low": 0.30,
    "medium": 0.60,
    "high": 0.80,
}

Triple = tuple[str, str, str]
State = FrozenSet[Triple]


def qualified_triples_by_course(data: ScenarioData) -> dict[str, list[Triple]]:
    """Build and cache course -> qualified triples for large SA runs."""
    cached = getattr(data, "_sa_qualified_triples_by_course", None)
    if cached is not None:
        return cached

    by_course: dict[str, list[Triple]] = {}
    for triple in data.qualified_triples:
        by_course.setdefault(triple[1], []).append(triple)
    setattr(data, "_sa_qualified_triples_by_course", by_course)
    return by_course


def qualified_triples_tuple(data: ScenarioData) -> tuple[Triple, ...]:
    """Build and cache a tuple version for fast random.choice calls."""
    cached = getattr(data, "_sa_qualified_triples_tuple", None)
    if cached is not None:
        return cached

    triples = tuple(data.qualified_triples)
    setattr(data, "_sa_qualified_triples_tuple", triples)
    return triples


@dataclass
class AnnealingStep:
    """One SA proposal, captured for later visualization."""

    iteration: int
    level: int
    temperature: float
    current_state: State
    current_cost: float
    current_objective: float
    current_violations: int
    proposed_state: State
    proposed_cost: float
    delta: float
    accept_probability: float
    accepted: bool
    best_state: State
    best_objective: float | None
    best_violations: int


@dataclass
class AnnealingStats:
    """Summary metrics for comparing temperature choices."""

    starting_temperature: str
    target_bad_move_acceptance: float
    initial_cost: float
    final_cost: float
    best_cost: float
    acceptance_rate: float
    feasible_states_visited: int
    final_constraint_violations: int
    accepted_moves: int
    proposed_moves: int


def generate_random_state(data: ScenarioData, rng: random.Random) -> State:
    """
    Generate a randomized starting state.

    A state is a frozen set of (professor_id, course_id, timeslot_id) triples.

    The real_solve1.py IP constraints you are trying to respect:
      * every required course is covered exactly once,
      * every optional course is covered at most once,
      * no professor exceeds max_load,
      * no professor teaches two courses in the same timeslot.

    This starter only assigns required courses. It shuffles the required
    courses and each course's candidate triples so the same professor does not
    get first shot every time. It tries to avoid overloads and double-booking;
    if a course has no conflict-free option, it still assigns a random
    qualified triple and lets the cost penalty handle the violation.
    """
    selected: set[Triple] = set()
    prof_loads: Counter[str] = Counter()
    prof_times: set[tuple[str, str]] = set()

    required_course_ids = list(data.required_course_ids)
    rng.shuffle(required_course_ids)

    triples_by_course = qualified_triples_by_course(data)

    for course_id in required_course_ids:
        candidates = triples_by_course.get(course_id, [])
        if not candidates:
            continue

        shuffled = list(candidates)
        rng.shuffle(shuffled)

        chosen = None
        for prof_id, candidate_course_id, ts_id in shuffled:
            prof = data.ta_by_id[prof_id]
            if prof_loads[prof_id] >= prof.max_load:
                continue
            if (prof_id, ts_id) in prof_times:
                continue
            chosen = (prof_id, candidate_course_id, ts_id)
            break

        if chosen is None:
            chosen = rng.choice(candidates)

        prof_id, _candidate_course_id, ts_id = chosen
        selected.add(chosen)
        prof_loads[prof_id] += 1
        prof_times.add((prof_id, ts_id))

    return frozenset(selected)


def cost(state: State, data: ScenarioData, temperature: float) -> float:
    """
    # E(x) = Sum(w_i x_i) + (1/t) num missing required classes + num double booked professor/class timeslots
    
    # (100 / t) chosen so as the temperature decreases, the penalty for not having required classes or 
    # having double booked increases
    """

    summary = constraint_violation_summary(state, data) 
    # calls a helper function which gives a count of how many missing required classes 
    # and how many professors are double booked for the same timeslot
    violations = summary["missing_required"] + summary["double_bookings"]
    return -objective_score(state, data) + (100 / (temperature)) * violations


def select_starting_temperature(
    data: ScenarioData,
    rng: random.Random,
    target_bad_move_acceptance: float,
    deadline: float | None = None,
) -> float:
    """
    Choose the starting temperature from a target bad-move acceptance rate.

    Temperature controls how often worse moves are accepted early. This
    estimates a "typical bad move" by sampling small random replacement moves
    around random starting states, then solves:

        target_bad_move_acceptance = exp(-typical_bad_delta / T)

    Since your final cost uses a 1 / T penalty, estimating deltas is a little
    circular. For the temperature probe, we use a reference penalty weight of
    1.0, equivalent to thinking "what would the move cost at T = 1?"
    """
    bad_deltas: list[float] = []
    n_samples = max(100, min(1000, 2 * len(data.qualified_triples)))

    for _ in range(n_samples):
        if deadline is not None and time.time() >= deadline:
            break

        state = generate_random_state(data, rng)
        neighbor = temperature_probe_neighbor(state, data, rng)
        delta = temperature_probe_cost(neighbor, data) - temperature_probe_cost(state, data)
        if delta > 0:
            bad_deltas.append(delta)

    if not bad_deltas:
        return 1.0

    bad_deltas.sort()
    typical_bad_delta = bad_deltas[len(bad_deltas) // 2]
    return temperature_for_acceptance_target(typical_bad_delta, target_bad_move_acceptance)


def states_per_temperature(data: ScenarioData) -> int:

    k = 10 # linear constant

    return max(1, int(k * math.sqrt(len(data.qualified_triples))))

def choose_cooling_rate(data: ScenarioData) -> float:
    """
    This scaffold uses geometric cooling:

        temperature = temperature * cooling_rate
    """

    cooling_rate = 0.98
    return cooling_rate

def propose_neighbor(state: State, data: ScenarioData, rng: random.Random) -> State:
    """
    Propose a neighboring state.

    This move set allows the two violations that the cost function penalizes:
    missing required courses and professor-timeslot double bookings. It tries
    to avoid professor overloads and repeated assignments for the same course,
    since those are not part of the current cost function.
    """
    next_state = set(state)
    if not data.qualified_triples:
        return state

    course_counts = Counter(course_id for _prof_id, course_id, _ts_id in next_state)
    prof_loads = Counter(prof_id for prof_id, _course_id, _ts_id in next_state)
    triples_by_course = qualified_triples_by_course(data)

    def candidate_triples_for_course(course_id: str) -> list[Triple]:
        return triples_by_course.get(course_id, [])

    def can_add_without_overload(triple: Triple, removed: set[Triple] | None = None) -> bool:
        removed = removed or set()
        prof_id = triple[0]
        removed_for_prof = sum(1 for old in removed if old[0] == prof_id)
        load_after_removal = prof_loads[prof_id] - removed_for_prof
        already_kept = triple in next_state and triple not in removed
        load_after_add = load_after_removal + (0 if already_kept else 1)
        return load_after_add <= data.ta_by_id[prof_id].max_load

    def replace_course_assignment(course_id: str, candidates: list[Triple]) -> bool:
        existing = {triple for triple in next_state if triple[1] == course_id}
        if not existing:
            return False

        usable = [
            triple
            for triple in candidates
            if triple in data.qualified_triples
            and triple not in next_state
            and can_add_without_overload(triple, removed=existing)
        ]
        if not usable:
            return False

        next_state.difference_update(existing)
        next_state.add(rng.choice(usable))
        return True

    def assigned_courses() -> list[str]:
        return list({course_id for _prof_id, course_id, _ts_id in next_state})

    def assigned_required_courses() -> list[str]:
        return [
            course_id
            for course_id in assigned_courses()
            if course_id in data.required_course_ids
        ]

    def assigned_optional_triples() -> list[Triple]:
        return [
            triple
            for triple in next_state
            if triple[1] in data.optional_course_ids
        ]

    def add_course(course_id: str) -> bool:
        if course_counts[course_id] > 0:
            return False
        candidates = [
            triple
            for triple in candidate_triples_for_course(course_id)
            if triple not in next_state and can_add_without_overload(triple)
        ]
        if not candidates:
            return False
        next_state.add(rng.choice(candidates))
        return True

    move_type = rng.choice([
        "change_professor",
        "change_timeslot",
        "replace_assignment",
        "add_optional",
        "remove_optional",
        #"swap_required_timeslots",
        "remove_required",
        "add_required",
    ])

    if move_type == "change_professor":
        courses = assigned_courses()
        if not courses:
            return state
        course_id = rng.choice(courses)
        existing = [triple for triple in next_state if triple[1] == course_id]
        if not existing:
            return state
        old_prof_id, _old_course_id, old_ts_id = rng.choice(existing)
        same_time_candidates = [
            triple
            for triple in candidate_triples_for_course(course_id)
            if triple[0] != old_prof_id and triple[2] == old_ts_id
        ]
        if replace_course_assignment(course_id, same_time_candidates):
            return frozenset(next_state)
        if replace_course_assignment(course_id, candidate_triples_for_course(course_id)):
            return frozenset(next_state)
        return state

    if move_type == "change_timeslot":
        courses = assigned_courses()
        if not courses:
            return state
        course_id = rng.choice(courses)
        existing = [triple for triple in next_state if triple[1] == course_id]
        if not existing:
            return state
        old_prof_id, _old_course_id, old_ts_id = rng.choice(existing)
        same_prof_candidates = [
            triple
            for triple in candidate_triples_for_course(course_id)
            if triple[0] == old_prof_id and triple[2] != old_ts_id
        ]
        if replace_course_assignment(course_id, same_prof_candidates):
            return frozenset(next_state)
        if replace_course_assignment(course_id, candidate_triples_for_course(course_id)):
            return frozenset(next_state)
        return state

    if move_type == "replace_assignment":
        courses = assigned_courses()
        if not courses:
            return state
        course_id = rng.choice(courses)
        if replace_course_assignment(course_id, candidate_triples_for_course(course_id)):
            return frozenset(next_state)
        return state

    if move_type == "add_optional":
        available_optional = [
            course_id
            for course_id in data.optional_course_ids
            if course_counts[course_id] == 0
        ]
        rng.shuffle(available_optional)
        for course_id in available_optional:
            if add_course(course_id):
                return frozenset(next_state)
        return state

    if move_type == "remove_optional":
        optional_triples = assigned_optional_triples()
        if not optional_triples:
            return state
        next_state.remove(rng.choice(optional_triples))
        return frozenset(next_state)

    if move_type == "swap_required_timeslots":
        required_courses = assigned_required_courses()
        if len(required_courses) < 2:
            return state
        course_a, course_b = rng.sample(required_courses, 2)
        existing_a = [triple for triple in next_state if triple[1] == course_a]
        existing_b = [triple for triple in next_state if triple[1] == course_b]
        if not existing_a or not existing_b:
            return state
        triple_a = rng.choice(existing_a)
        triple_b = rng.choice(existing_b)
        new_a = (triple_a[0], triple_a[1], triple_b[2])
        new_b = (triple_b[0], triple_b[1], triple_a[2])
        removed = {triple_a, triple_b}
        if (
            new_a in data.qualified_triples
            and new_b in data.qualified_triples
            and new_a != new_b
            and can_add_without_overload(new_a, removed=removed)
            and can_add_without_overload(new_b, removed=removed)
        ):
            next_state.difference_update(removed)
            next_state.add(new_a)
            next_state.add(new_b)
            return frozenset(next_state)
        return state

    if move_type == "remove_required":
        required_triples = [
            triple
            for triple in next_state
            if triple[1] in data.required_course_ids
        ]
        if not required_triples:
            return state
        next_state.remove(rng.choice(required_triples))
        return frozenset(next_state)

    if move_type == "add_required":
        missing_required = [
            course_id
            for course_id in data.required_course_ids
            if course_counts[course_id] == 0
        ]
        rng.shuffle(missing_required)
        for course_id in missing_required:
            if add_course(course_id):
                return frozenset(next_state)
        return state

    return state


def objective_score(state: State, data: ScenarioData) -> float:
    """Compute the same objective contribution used by real_solve1.py."""
    assignment_score = _real_solve1_attr("assignment_score")
    return sum(assignment_score(prof_id, course_id, ts_id, data) for prof_id, course_id, ts_id in state)


def state_to_solution(
    state: State,
    data: ScenarioData,
    solver_time_sec: float,
    notes: str = "",
) -> Solution:
    """Convert an SA state into the shared Solution object used by validators."""
    return Solution(
        status=STATUS_FEASIBLE,
        objective=objective_score(state, data),
        selected_triples=sorted(state, key=lambda t: (t[1], t[2], t[0])),
        solver_time_sec=solver_time_sec,
        notes=notes,
    )


def run_simulated_annealing(
    data: ScenarioData,
    seed: int,
    levels: int,
    starting_temperature: str,
    target_bad_move_acceptance: float,
    history: list[AnnealingStep] | None = None,
    timeout_sec: float | None = None,
) -> tuple[Solution, AnnealingStats]:
    """
    Run the generic SA loop around your TODO functions.

    This part is intentionally provided so you can focus on the five modeling
    decisions from your plan, plus the required neighbor function.
    """
    rng = random.Random(seed)
    start = time.time()
    deadline = start + timeout_sec if timeout_sec is not None else None

    current = generate_random_state(data, rng)

    temperature = select_starting_temperature(
        data,
        rng,
        target_bad_move_acceptance,
        deadline=deadline,
    )
    moves_per_temperature = states_per_temperature(data)
    cooling_rate = choose_cooling_rate(data)

    if temperature <= 0:
        raise ValueError("select_starting_temperature must return a positive number")
    if moves_per_temperature <= 0:
        raise ValueError("states_per_temperature must return a positive integer")
    if not 0 < cooling_rate < 1:
        raise ValueError("choose_cooling_rate must return a value strictly between 0 and 1")

    best = current
    initial_cost = cost(current, data, temperature)
    best_cost = initial_cost
    best_objective = objective_score(current, data) if is_feasible_state(current, data) else None

    accepted = 0
    proposed = 0
    feasible_states_visited = 1 if is_feasible_state(current, data) else 0
    iteration = 0
    timed_out = deadline is not None and time.time() >= deadline

    for level in range(levels):
        if timed_out:
            break
        for _ in range(moves_per_temperature):
            if deadline is not None and time.time() >= deadline:
                timed_out = True
                break

            proposed += 1
            current_cost = cost(current, data, temperature)
            candidate = propose_neighbor(current, data, rng)
            candidate_cost = cost(candidate, data, temperature)
            delta = candidate_cost - current_cost
            accept_probability = metropolis_accept_probability(delta, temperature)
            accept = rng.random() < accept_probability

            if accept:
                accepted += 1
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                if is_feasible_state(candidate, data):
                    feasible_states_visited += 1
                candidate_objective = objective_score(candidate, data)
                if is_feasible_state(candidate, data):
                    if best_objective is None or candidate_objective > best_objective:
                        best = candidate
                        best_objective = candidate_objective
                elif best_objective is None:
                    best = candidate

            if history is not None:
                history.append(
                    AnnealingStep(
                        iteration=iteration,
                        level=level,
                        temperature=temperature,
                        current_state=current,
                        current_cost=current_cost,
                        current_objective=objective_score(current, data),
                        current_violations=total_constraint_violations(current, data),
                        proposed_state=candidate,
                        proposed_cost=candidate_cost,
                        delta=delta,
                        accept_probability=accept_probability,
                        accepted=accept,
                        best_state=best,
                        best_objective=best_objective,
                        best_violations=total_constraint_violations(best, data),
                    )
                )
            iteration += 1

        if timed_out:
            break
        temperature *= cooling_rate

    elapsed = time.time() - start
    final_cost = cost(current, data, temperature)
    final_violations = total_constraint_violations(current, data)
    stats = AnnealingStats(
        starting_temperature=starting_temperature,
        target_bad_move_acceptance=target_bad_move_acceptance,
        initial_cost=initial_cost,
        final_cost=final_cost,
        best_cost=best_cost,
        acceptance_rate=accepted / proposed if proposed else 0.0,
        feasible_states_visited=feasible_states_visited,
        final_constraint_violations=final_violations,
        accepted_moves=accepted,
        proposed_moves=proposed,
    )
    solution = state_to_solution(
        best,
        data,
        solver_time_sec=elapsed,
        notes=(
            f"SA accepted {accepted}/{proposed} proposed moves"
            + (f"; timed out after {timeout_sec:.1f}s" if timed_out else "")
        ),
    )
    return solution, stats


def metropolis_accept_probability(delta: float, temperature: float) -> float:
    """
    Acceptance rule: p = min(1, exp(-delta / T)).

    The delta <= 0 branch is the same formula, but avoids computing exp() on
    a huge positive number when a move is much better than the current state.
    """
    if delta <= 0:
        return 1.0
    return min(1.0, math.exp(-delta / temperature))


def temperature_for_acceptance_target(typical_bad_delta: float, target_acceptance: float) -> float:
    """
    Convert a typical uphill delta into T for p = exp(-delta / T).

    Example targets:
      * 0.30 accepts about 30% of typical bad moves.
      * 0.60 accepts about 60% of typical bad moves.
      * 0.80 accepts about 80% of typical bad moves.
    """
    if typical_bad_delta <= 0:
        raise ValueError("typical_bad_delta must be positive")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must be strictly between 0 and 1")
    return -typical_bad_delta / math.log(target_acceptance)


def temperature_probe_cost(state: State, data: ScenarioData) -> float:
    """
    Lightweight reference cost for estimating the starting temperature.

    This intentionally does not call your real cost(...) TODO. It mirrors the
    shape you want at reference T = 1:

        -objective + violations
    """
    return -objective_score(state, data) + total_constraint_violations(state, data)


def temperature_probe_neighbor(state: State, data: ScenarioData, rng: random.Random) -> State:
    """
    Small random replacement move used only for estimating typical bad deltas.

    The real SA run still uses your propose_neighbor(...) implementation.
    """
    qualified_triples = qualified_triples_tuple(data)
    if not qualified_triples:
        return state

    next_state = set(state)
    if not next_state:
        next_state.add(rng.choice(qualified_triples))
        return frozenset(next_state)

    removed = rng.choice(tuple(next_state))
    next_state.remove(removed)

    same_course_candidates = [
        triple
        for triple in qualified_triples_by_course(data).get(removed[1], [])
        if triple != removed
    ]
    if same_course_candidates:
        next_state.add(rng.choice(same_course_candidates))
    else:
        next_state.add(rng.choice(qualified_triples))

    return frozenset(next_state)


def is_feasible_state(state: State, data: ScenarioData) -> bool:
    """Return True when the state satisfies the real_solve1.py constraints."""
    return total_constraint_violations(state, data) == 0


def total_constraint_violations(state: State, data: ScenarioData) -> int:
    """Scalar violation count useful for penalties, titles, and plots."""
    return sum(constraint_violation_summary(state, data).values())


def constraint_violation_summary(state: State, data: ScenarioData) -> dict[str, int]:
    """
    Count real_solve1.py constraint violations in a way that is useful for SA.

    This helper is intentionally available for your cost function. You still
    decide how much each violation type should matter.
    """
    course_counts: Counter[str] = Counter()
    prof_counts: Counter[str] = Counter()
    prof_time_counts: Counter[tuple[str, str]] = Counter()
    invalid_triples = 0

    for prof_id, course_id, ts_id in state:
        if (prof_id, course_id, ts_id) not in data.qualified_triples:
            invalid_triples += 1
        course_counts[course_id] += 1
        prof_counts[prof_id] += 1
        prof_time_counts[(prof_id, ts_id)] += 1

    missing_required = sum(
        max(0, 1 - course_counts[course_id])
        for course_id in data.required_course_ids
    )
    extra_required = sum(
        max(0, course_counts[course_id] - 1)
        for course_id in data.required_course_ids
    )
    overcovered_optional = sum(
        max(0, course_counts[course_id] - 1)
        for course_id in data.optional_course_ids
    )
    overloads = sum(
        max(0, prof_counts[prof.ta_id] - prof.max_load)
        for prof in data.tas
    )
    double_bookings = sum(
        max(0, count - 1)
        for count in prof_time_counts.values()
    )

    return {
        "invalid_triples": invalid_triples,
        "missing_required": missing_required,
        "extra_required": extra_required,
        "overcovered_optional": overcovered_optional,
        "overloads": overloads,
        "double_bookings": double_bookings,
    }


def animate_annealing(
    history: list[AnnealingStep],
    data: ScenarioData,
    args: argparse.Namespace,
    ip_solution: Solution | None = None,
) -> None:
    """
    Animate the SA run in the same spirit as sa_playground/demo_tsp.py.

    Left: current assignment matrix. Each dot is one selected
    (professor, course, timeslot) triple.

    Top right: objective and violation count over time.
    Bottom right: temperature and uphill acceptance probability.
    """
    if not history:
        print("\nno annealing history to visualize")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    courses = sorted(data.courses, key=lambda c: (not c.is_required, c.course_code, c.course_id))
    professors = sorted(data.tas, key=lambda p: (p.name, p.ta_id))
    timeslots = sorted(
        data.timeslots,
        key=lambda t: (t.day, t.start_time, t.end_time, t.timeslot_id),
    )

    course_index = {course.course_id: idx for idx, course in enumerate(courses)}
    professor_index = {prof.ta_id: idx for idx, prof in enumerate(professors)}
    timeslot_index = {ts.timeslot_id: idx for idx, ts in enumerate(timeslots)}

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax_assign = fig.add_subplot(gs[:, 0])
    ax_obj = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[1, 1])

    for idx, course in enumerate(courses):
        if course.is_required:
            ax_assign.axvspan(idx - 0.5, idx + 0.5, color="tab:green", alpha=0.08)

    current_scatter = ax_assign.scatter(
        [],
        [],
        c=[],
        cmap="tab20",
        vmin=-0.5,
        vmax=max(len(timeslots) - 0.5, 0.5),
        s=42,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
        label="current",
    )
    best_scatter = ax_assign.scatter(
        [],
        [],
        facecolors="none",
        edgecolors="tab:green",
        marker="s",
        s=90,
        linewidth=1.0,
        label="best feasible",
    )
    ax_assign.set_xlim(-0.5, len(courses) - 0.5)
    ax_assign.set_ylim(-0.5, len(professors) - 0.5)
    ax_assign.set_xlabel("course")
    ax_assign.set_ylabel("professor")
    ax_assign.set_xticks(range(len(courses)))
    ax_assign.set_xticklabels([c.course_code for c in courses], rotation=90, fontsize=6)
    ax_assign.set_yticks(range(len(professors)))
    ax_assign.set_yticklabels([p.name for p in professors], fontsize=5)
    ax_assign.grid(color="0.92", linewidth=0.4)
    ax_assign.legend(loc="upper right")
    cbar = fig.colorbar(current_scatter, ax=ax_assign, fraction=0.026, pad=0.02)
    cbar.set_label("timeslot index")

    iters = [step.iteration for step in history]
    current_objectives = [step.current_objective for step in history]
    best_objectives = [
        step.best_objective if step.best_objective is not None else float("nan")
        for step in history
    ]
    violations = [step.current_violations for step in history]
    temps = [step.temperature for step in history]
    uphill_acceptance = [
        step.accept_probability if step.delta > 0 else float("nan")
        for step in history
    ]

    ax_obj.plot(iters, current_objectives, color="tab:blue", linewidth=0.6, alpha=0.35)
    current_obj_line, = ax_obj.plot([], [], color="tab:blue", linewidth=1.0, label="current objective")
    best_obj_line, = ax_obj.plot([], [], color="tab:red", linewidth=1.4, linestyle="--", label="best feasible")
    if ip_solution and ip_solution.objective is not None:
        ax_obj.axhline(ip_solution.objective, color="tab:green", linewidth=1.0, linestyle=":", label="IP optimum")
    ax_obj.set_title("Objective + constraint violations")
    ax_obj.set_ylabel("objective")
    finite_best = [v for v in best_objectives if not math.isnan(v)]
    objective_values = current_objectives + finite_best
    if ip_solution and ip_solution.objective is not None:
        objective_values.append(ip_solution.objective)
    obj_min = min(objective_values)
    obj_max = max(objective_values)
    obj_pad = max((obj_max - obj_min) * 0.08, 1.0)
    ax_obj.set_ylim(obj_min - obj_pad, obj_max + obj_pad)
    ax_obj.set_xlim(0, max(iters) if iters else 1)

    ax_viol = ax_obj.twinx()
    violation_line, = ax_viol.plot([], [], color="tab:purple", linewidth=1.0, alpha=0.75, label="violations")
    ax_viol.set_ylabel("violations")
    ax_viol.set_ylim(0, max(violations + [1]) + 1)

    lines = ax_obj.get_lines() + ax_viol.get_lines()
    labels = [line.get_label() for line in lines]
    ax_obj.legend(lines, labels, loc="upper right")

    ax_temp.set_yscale("log")
    ax_temp.set_xlim(0, max(iters) if iters else 1)
    positive_temps = [t for t in temps if t > 0]
    ax_temp.set_ylim(max(min(positive_temps), 1e-12), max(positive_temps) * 1.1)
    temp_line, = ax_temp.plot([], [], color="tab:orange", label="T")
    ax_prob = ax_temp.twinx()
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("P(accept worse)")
    p_scatter = ax_prob.scatter([], [], s=4, color="tab:purple", alpha=0.4)
    ax_temp.set_title("Temperature (log) + acceptance prob. of uphill moves")
    ax_temp.set_xlabel("iteration")
    ax_temp.legend(loc="upper right")

    n_frames = min(args.frames, len(history))
    frame_idx = np.linspace(0, len(history) - 1, n_frames).astype(int)

    def scatter_data(state: State):
        xs: list[int] = []
        ys: list[int] = []
        colors: list[int] = []
        for prof_id, course_id, ts_id in state:
            if course_id not in course_index or prof_id not in professor_index:
                continue
            xs.append(course_index[course_id])
            ys.append(professor_index[prof_id])
            colors.append(timeslot_index.get(ts_id, 0))
        if not xs:
            return np.empty((0, 2)), np.array([])
        return np.column_stack([xs, ys]), np.array(colors)

    def update(frame_i: int):
        upto = frame_idx[frame_i]
        step = history[upto]

        current_offsets, current_colors = scatter_data(step.current_state)
        current_scatter.set_offsets(current_offsets)
        current_scatter.set_array(current_colors)

        if step.best_objective is None:
            best_offsets = np.empty((0, 2))
        else:
            best_offsets, _ = scatter_data(step.best_state)
        best_scatter.set_offsets(best_offsets)

        current_obj_line.set_data(iters[:upto + 1], current_objectives[:upto + 1])
        best_obj_line.set_data(iters[:upto + 1], best_objectives[:upto + 1])
        violation_line.set_data(iters[:upto + 1], violations[:upto + 1])
        temp_line.set_data(iters[:upto + 1], temps[:upto + 1])

        points = [
            (iters[i], uphill_acceptance[i])
            for i in range(upto + 1)
            if not math.isnan(uphill_acceptance[i])
        ]
        if points:
            p_scatter.set_offsets(np.array(points))

        best_label = "none" if step.best_objective is None else f"{step.best_objective:.3f}"
        ax_assign.set_title(
            f"iter {step.iteration}/{history[-1].iteration}   "
            f"T = {step.temperature:.4g}   "
            f"current obj = {step.current_objective:.3f}   "
            f"best feasible = {best_label}   "
            f"violations = {step.current_violations}"
        )
        return (
            current_scatter,
            best_scatter,
            current_obj_line,
            best_obj_line,
            violation_line,
            temp_line,
            p_scatter,
        )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=args.interval, blit=False, repeat=False)
    plt.tight_layout()

    if args.save:
        anim.save(args.save, fps=30, dpi=120)
        print(f"saved animation to {args.save}")
    else:
        plt.show()


def run_ip_baseline(data: ScenarioData) -> Solution:
    """Run the exact IP baseline from real_solve1.py."""
    solve_ip = _real_solve1_attr("solve")
    return solve_ip(data)


def _real_solve1_attr(name: str) -> Callable:
    """
    Import real_solve1 lazily so this scaffold can still compile even if the
    local machine has not finished setting up Gurobi.
    """
    try:
        module = importlib.import_module("real_solve1")
    except Exception as exc:
        raise RuntimeError(
            "Could not import real_solve1.py. The IP comparison and objective "
            "helper need that module, including its Gurobi dependency."
        ) from exc
    return getattr(module, name)


def print_solution_report(label: str, solution: Solution, data: ScenarioData) -> ValidationResult:
    """Print objective, status, validation messages, and timing for one solution."""
    validation = validate2(solution, data)

    print(f"\n=== {label} ===")
    print(f"  status:    {solution.status}")
    print(f"  objective: {format_optional_float(solution.objective)}")
    print(f"  selected:  {len(solution.selected_triples)} triples")
    print(f"  time:      {format_optional_float(solution.solver_time_sec)}s")
    if solution.notes:
        print(f"  notes:     {solution.notes}")

    for error in validation.errors:
        print(f"  FAIL: {error}")
    for warning in validation.warnings:
        print(f"  warn: {warning}")
    print(f"  valid:     {'yes' if validation.ok else 'no'}")

    return validation


def print_gap_report(ip_solution: Solution | None, sa_solution: Solution | None) -> None:
    """Print SA-vs-IP gap when both objectives are available."""
    if ip_solution is None or sa_solution is None:
        return
    if ip_solution.objective is None or sa_solution.objective is None:
        print("\n=== COMPARISON ===")
        print("  gap unavailable because at least one objective is missing")
        return

    absolute_gap = ip_solution.objective - sa_solution.objective
    relative_gap = absolute_gap / ip_solution.objective if ip_solution.objective else float("nan")
    achieved = sa_solution.objective / ip_solution.objective if ip_solution.objective else float("nan")

    print("\n=== COMPARISON ===")
    print(f"  absolute gap:       {absolute_gap:.4f}")
    print(f"  relative gap:       {100 * relative_gap:.2f}%")
    print(f"  SA percent of IP:   {100 * achieved:.2f}%")


def print_annealing_stats(stats: AnnealingStats) -> None:
    """Print the temperature-comparison metrics requested for SA experiments."""
    print("\n=== ANNEALING STATS ===")
    print(f"  starting temperature:       {stats.starting_temperature}")
    print(f"  target bad-move acceptance: {100 * stats.target_bad_move_acceptance:.0f}%")
    print(f"  initial cost:               {stats.initial_cost:.4f}")
    print(f"  final cost:                 {stats.final_cost:.4f}")
    print(f"  best cost found:            {stats.best_cost:.4f}")
    print(f"  acceptance rate:            {100 * stats.acceptance_rate:.2f}%")
    print(f"  accepted moves:             {stats.accepted_moves}/{stats.proposed_moves}")
    print(f"  feasible states visited:    {stats.feasible_states_visited}")
    print(f"  final constraint violations:{stats.final_constraint_violations:>5d}")


def print_temperature_comparison(stats_by_profile: list[AnnealingStats]) -> None:
    """Print one compact table when --starting-temperature all is used."""
    if not stats_by_profile:
        return

    print("\n=== TEMPERATURE COMPARISON ===")
    print(
        "  profile   target   initial_cost   final_cost   best_cost   "
        "accept_rate   feasible   final_viol"
    )
    for stats in stats_by_profile:
        print(
            f"  {stats.starting_temperature:7s} "
            f"{100 * stats.target_bad_move_acceptance:6.0f}% "
            f"{stats.initial_cost:14.4f} "
            f"{stats.final_cost:12.4f} "
            f"{stats.best_cost:11.4f} "
            f"{100 * stats.acceptance_rate:10.2f}% "
            f"{stats.feasible_states_visited:10d} "
            f"{stats.final_constraint_violations:10d}"
        )


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learning scaffold for simulated annealing on real_solve1.py."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"directory containing generated UW scenarios (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"scenario directory name (default: {DEFAULT_SCENARIO})",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed for SA")
    parser.add_argument(
        "--levels",
        type=int,
        default=200,
        help="number of temperature levels in the SA run",
    )
    parser.add_argument(
        "--starting-temperature",
        "--temperature-profile",
        dest="starting_temperature",
        choices=["low", "medium", "high", "all"],
        default="medium",
        help=(
            "starting temperature: low=30%%, medium=60%%, high=80%% "
            "acceptance of typical bad moves"
        ),
    )
    parser.add_argument(
        "--skip-ip",
        action="store_true",
        help="skip the Gurobi IP baseline and run only the SA scaffold",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="run SA without opening the matplotlib animation window",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=400,
        help="animation frames; history is subsampled like demo_tsp.py",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=25,
        help="milliseconds between animation frames",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="optional path to save animation as .mp4 or .gif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.data_dir) / args.scenario

    print(f"loading scenario: {scenario_dir}")
    data = load_scenario(scenario_dir)
    print(
        f"  professors: {len(data.tas)}   courses: {len(data.courses)}   "
        f"timeslots: {len(data.timeslots)}   qualified triples: {len(data.qualified_triples)}"
    )

    ip_solution: Solution | None = None
    if not args.skip_ip:
        print("\nrunning exact IP baseline from real_solve1.py...")
        try:
            ip_solution = run_ip_baseline(data)
            print_solution_report("IP BASELINE", ip_solution, data)
        except Exception:
            ip_solution = Solution(
                status=STATUS_ERROR,
                notes="IP baseline raised an exception; see traceback above.",
            )
            print("  IP BASELINE RAISED:")
            traceback.print_exc()

    starting_temperatures = (
        list(STARTING_TEMPERATURE_TARGETS)
        if args.starting_temperature == "all"
        else [args.starting_temperature]
    )
    should_visualize = (
        not args.no_visualize or args.save is not None
    ) and len(starting_temperatures) == 1
    if args.starting_temperature == "all" and (not args.no_visualize or args.save is not None):
        print("\nvisualization is skipped for --starting-temperature all; rerun one setting to animate it")

    all_stats: list[AnnealingStats] = []
    last_history: list[AnnealingStep] | None = None

    for starting_temperature in starting_temperatures:
        target_acceptance = STARTING_TEMPERATURE_TARGETS[starting_temperature]
        print(
            f"\nrunning simulated annealing scaffold "
            f"(starting temperature {starting_temperature}, "
            f"target bad-move acceptance {100 * target_acceptance:.0f}%)..."
        )
        history: list[AnnealingStep] | None = [] if should_visualize else None
        try:
            sa_solution, stats = run_simulated_annealing(
                data,
                seed=args.seed,
                levels=args.levels,
                starting_temperature=starting_temperature,
                target_bad_move_acceptance=target_acceptance,
                history=history,
            )
        except NotImplementedError as exc:
            print("\n=== SA TODO ===")
            print(f"  {exc}")
            print("  Fill in that function, then rerun this file.")
            return

        print_solution_report(
            f"SIMULATED ANNEALING ({starting_temperature})",
            sa_solution,
            data,
        )
        print_annealing_stats(stats)
        print_gap_report(ip_solution, sa_solution)

        all_stats.append(stats)
        last_history = history

    if args.starting_temperature == "all":
        print_temperature_comparison(all_stats)

    if should_visualize and last_history is not None:
        animate_annealing(last_history, data, args, ip_solution=ip_solution)


if __name__ == "__main__":
    main()
