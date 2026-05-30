"""
Core simulated annealing algorithm.

Simulated annealing is a stochastic local search method. At each step it
proposes a small perturbation of the current state (a "neighbor"). If the
neighbor is better, it always moves there. If the neighbor is worse, it
moves there with probability exp(-delta / T), where T is the "temperature"
that gradually cools toward zero.

High T early on => bad moves are often accepted => the search explores.
Low T later     => bad moves are rarely accepted => the search exploits.

The whole point: pure greedy descent gets trapped in the first local minimum
it finds. SA can climb out of shallow valleys when it is hot, then settle
into a deep one when it is cold.

This file is intentionally tiny and generic — the problem-specific pieces
(initial state, neighbor function, energy function) are supplied by the
caller. See demo_function.py and demo_tsp.py for two concrete uses.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Generic, List, TypeVar

State = TypeVar("State")


@dataclass
class Step(Generic[State]):
    """One step of the SA run, captured for later inspection / animation."""
    iteration: int
    temperature: float
    current_state: State
    current_energy: float
    proposed_state: State
    proposed_energy: float
    delta: float
    accept_probability: float   # 1.0 if proposed is better, else exp(-delta/T)
    accepted: bool
    best_energy_so_far: float


@dataclass
class Result(Generic[State]):
    best_state: State
    best_energy: float
    history: List[Step] = field(default_factory=list)


def exponential_cooling(t0: float, alpha: float) -> Callable[[int], float]:
    """T(k) = t0 * alpha^k. Most common schedule. alpha typically 0.90-0.999."""
    return lambda k: t0 * (alpha ** k)


def linear_cooling(t0: float, t_min: float, steps: int) -> Callable[[int], float]:
    """T drops linearly from t0 to t_min over `steps` iterations."""
    slope = (t0 - t_min) / max(steps - 1, 1)
    return lambda k: max(t0 - slope * k, t_min)


def anneal(
    initial_state: State,
    energy: Callable[[State], float],
    neighbor: Callable[[State], State],
    schedule: Callable[[int], float],
    n_iterations: int,
    rng: random.Random | None = None,
    record_history: bool = True,
) -> Result[State]:
    """
    Run simulated annealing.

    Parameters
    ----------
    initial_state : any
        Where to start. SA does not need a smart initial state, but a
        reasonable one shortens the run.
    energy : state -> float
        The thing we minimize. (For maximization, return -score.)
    neighbor : state -> state
        Produce a small random perturbation of the input state.
        "Small" matters: if neighbors are too different, SA degenerates
        into random search; if too similar, it never escapes anything.
    schedule : iteration -> temperature
        Returns the temperature to use at iteration k. See cooling helpers
        above.
    n_iterations : int
        Total steps. More iterations => better solution, more compute.
    rng : random.Random
        Pass one in to get reproducible runs.
    record_history : bool
        If True, every step is stored for animation. Turn off for speed.
    """
    rng = rng or random.Random()

    current = initial_state
    current_e = energy(current)
    best = current
    best_e = current_e

    history: List[Step] = []

    for k in range(n_iterations):
        t = schedule(k)
        proposed = neighbor(current)
        proposed_e = energy(proposed)
        delta = proposed_e - current_e

        if delta <= 0:
            # proposed is at least as good — always accept
            accept_p = 1.0
            accepted = True
        else:
            # proposed is worse — accept with Metropolis probability
            # the colder it gets, the smaller this number becomes
            accept_p = math.exp(-delta / t) if t > 0 else 0.0
            accepted = rng.random() < accept_p

        if record_history:
            history.append(Step(
                iteration=k,
                temperature=t,
                current_state=current,
                current_energy=current_e,
                proposed_state=proposed,
                proposed_energy=proposed_e,
                delta=delta,
                accept_probability=accept_p,
                accepted=accepted,
                best_energy_so_far=best_e,
            ))

        if accepted:
            current = proposed
            current_e = proposed_e
            if current_e < best_e:
                best = current
                best_e = current_e

    return Result(best_state=best, best_energy=best_e, history=history)
