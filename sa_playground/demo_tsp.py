"""
Watch simulated annealing solve a small Traveling Salesman Problem.

This is the more "combinatorial" view of SA — the state is a permutation
of N cities, and "neighbor" means swapping a chunk of the tour. The
energy is the tour's total length. This is closer in flavor to your
TA / course assignment problem: discrete decisions, no gradient to
follow, lots of nearly-equivalent local optima.

What to look for:
  * Left panel: the cities and the current tour. Watch it tangle and
    untangle. Early on (high T) the tour stays messy because SA happily
    accepts bad swaps. As T drops, only good swaps stick and the tour
    smooths out.
  * Top right: tour length over time. Like in the function demo, the
    *current* length can spike — that's an accepted worse move buying
    long-term progress.
  * Bottom right: temperature and acceptance probability for worse moves.

Neighbor move: 2-opt reverse. Pick two indices i < j and reverse the
segment between them. This is the textbook TSP perturbation — small
enough to be informative, big enough to escape bad orderings.

Usage:
    python3 demo_tsp.py
    python3 demo_tsp.py --cities 50 --iters 8000 --alpha 0.9985
    python3 demo_tsp.py --save tsp.mp4
"""

from __future__ import annotations

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from sa import anneal, exponential_cooling


def tour_length(tour: tuple[int, ...], coords: np.ndarray) -> float:
    pts = coords[list(tour)]
    diffs = np.diff(pts, axis=0)
    seg = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    # close the loop
    seg += math.hypot(*(coords[tour[0]] - coords[tour[-1]]))
    return float(seg)


def two_opt_neighbor(rng: random.Random):
    """Reverse a random contiguous segment of the tour."""
    def f(tour):
        n = len(tour)
        i, j = sorted(rng.sample(range(n), 2))
        return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
    return f


def run(args):
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    coords = np_rng.uniform(0, 1, size=(args.cities, 2))

    initial = tuple(rng.sample(range(args.cities), args.cities))

    result = anneal(
        initial_state=initial,
        energy=lambda t: tour_length(t, coords),
        neighbor=two_opt_neighbor(rng),
        schedule=exponential_cooling(args.t0, args.alpha),
        n_iterations=args.iters,
        rng=rng,
    )

    print(f"cities: {args.cities}")
    print(f"initial tour length: {tour_length(initial, coords):.4f}")
    print(f"final best length:   {result.best_energy:.4f}")
    print(f"accepted {sum(1 for s in result.history if s.accepted)} / {len(result.history)} proposals")

    animate(result, coords, args)


def animate(result, coords, args):
    history = result.history

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax_tour = fig.add_subplot(gs[:, 0])
    ax_len = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[1, 1])

    ax_tour.scatter(coords[:, 0], coords[:, 1], color="black", s=25, zorder=3)
    for idx, (x, y) in enumerate(coords):
        ax_tour.annotate(str(idx), (x, y), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
    tour_line, = ax_tour.plot([], [], color="tab:blue",
                              linewidth=1.5, alpha=0.85)
    best_tour_line, = ax_tour.plot([], [], color="tab:green",
                                   linewidth=1, alpha=0.4, linestyle="--")
    ax_tour.set_xlim(-0.05, 1.05)
    ax_tour.set_ylim(-0.05, 1.05)
    ax_tour.set_aspect("equal")
    ax_tour.set_title("Current tour (blue) vs. best tour so far (green dashed)")

    iters = [s.iteration for s in history]
    cur_e = [s.current_energy for s in history]
    best_e = [s.best_energy_so_far for s in history]
    temps = [s.temperature for s in history]
    accept_p = [s.accept_probability if s.delta > 0 else float("nan")
                for s in history]

    ax_len.plot(iters, cur_e, color="tab:blue", linewidth=0.6, alpha=0.4)
    cur_len_line, = ax_len.plot([], [], color="tab:blue",
                                linewidth=1.0, label="current")
    best_len_line, = ax_len.plot([], [], color="tab:red",
                                 linewidth=1.5, linestyle="--", label="best")
    ax_len.set_title("Tour length over time")
    ax_len.set_ylabel("length")
    ax_len.legend(loc="upper right")
    ax_len.set_xlim(0, len(history))
    ax_len.set_ylim(min(best_e) * 0.95, max(cur_e) * 1.02)

    ax_temp.set_yscale("log")
    ax_temp.set_xlim(0, len(history))
    ax_temp.set_ylim(max(min(temps), 1e-6), max(temps) * 1.1)
    temp_line, = ax_temp.plot([], [], color="tab:orange", label="T")
    ax_p = ax_temp.twinx()
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("P(accept worse)")
    p_scatter = ax_p.scatter([], [], s=3, color="tab:purple", alpha=0.4)
    ax_temp.set_title("Temperature (log) + acceptance prob. of uphill moves")
    ax_temp.set_xlabel("iteration")
    ax_temp.legend(loc="upper right")

    n_frames = min(args.frames, len(history))
    frame_idx = np.linspace(0, len(history) - 1, n_frames).astype(int)

    # track best-tour-so-far permutation as we walk through frames
    best_tour = history[0].current_state
    best_so_far = history[0].current_energy

    def closed_xy(tour):
        idx = list(tour) + [tour[0]]
        return coords[idx, 0], coords[idx, 1]

    def update(frame_i):
        nonlocal best_tour, best_so_far
        upto = frame_idx[frame_i]
        for k in range(upto + 1):
            if history[k].accepted and history[k].current_energy < best_so_far:
                best_so_far = history[k].current_energy
                best_tour = history[k].current_state

        cur_tour = history[upto].current_state
        tour_line.set_data(*closed_xy(cur_tour))
        best_tour_line.set_data(*closed_xy(best_tour))

        cur_len_line.set_data(iters[:upto + 1], cur_e[:upto + 1])
        best_len_line.set_data(iters[:upto + 1], best_e[:upto + 1])
        temp_line.set_data(iters[:upto + 1], temps[:upto + 1])

        pts = [(iters[i], accept_p[i]) for i in range(upto + 1)
               if not math.isnan(accept_p[i])]
        if pts:
            p_scatter.set_offsets(np.array(pts))

        ax_tour.set_title(
            f"iter {upto}/{len(history) - 1}   "
            f"T = {history[upto].temperature:.4f}   "
            f"current = {history[upto].current_energy:.3f}   "
            f"best = {best_so_far:.3f}"
        )
        return tour_line, best_tour_line, cur_len_line, best_len_line, temp_line

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=args.interval, blit=False, repeat=False)
    plt.tight_layout()

    if args.save:
        anim.save(args.save, fps=30, dpi=120)
        print(f"saved animation to {args.save}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", type=int, default=80)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--t0", type=float, default=1.0,
                   help="initial temperature, roughly the scale of a bad swap")
    p.add_argument("--alpha", type=float, default=0.9995,
                   help="exponential cooling factor")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--interval", type=int, default=25)
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
