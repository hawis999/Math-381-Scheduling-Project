"""
Watch simulated annealing minimize a bumpy 2D function.

The "Rastrigin" function is a standard hard test case: it has one global
minimum at (0, 0) but dozens of evenly-spaced local minima all over the
plane. A pure greedy descent from a random start almost always gets
stuck in the wrong basin. SA can climb out as long as it is still hot.

What to look for in the animation:
  * Left panel: contour plot of the landscape. Yellow trail = accepted
    moves, red X = current position, green star = best-so-far. While T
    is high you'll see the trail wander uphill freely. As T cools the
    trail settles into one basin.
  * Top right: energy over time. Red dashed line = best-so-far. Notice
    that the *current* energy can go up — that is the whole point.
  * Bottom right: temperature (log scale) and acceptance probability of
    each proposed worse move. Early on, acceptance is near 1 even for
    big uphill jumps. Late, it is near 0.

Usage:
    python3 demo_function.py
    python3 demo_function.py --iters 4000 --t0 5.0 --alpha 0.997
    python3 demo_function.py --save run.mp4

Try cooling too fast (alpha=0.9) and watch SA get trapped, then try a
slow cool (alpha=0.999) and watch it find the global minimum.
"""

from __future__ import annotations

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from sa import anneal, exponential_cooling


def rastrigin(x: float, y: float) -> float:
    """
    Rastrigin function. Global min = 0 at (0, 0). Domain ~ [-5.12, 5.12]^2.
    Lots of local minima on a regular grid, all with comparable depth, so
    a greedy method can't tell which one to head for.
    """
    A = 10.0
    return (
        2 * A
        + (x * x - A * math.cos(2 * math.pi * x))
        + (y * y - A * math.cos(2 * math.pi * y))
    )


def gaussian_neighbor(step_size: float, rng: random.Random):
    """
    Neighbor = current point plus a small Gaussian nudge in each dimension.
    The step size controls how far we look. Too small => SA can't escape
    local basins even when hot. Too large => SA is basically random search.
    """
    def f(state):
        x, y = state
        return (x + rng.gauss(0, step_size), y + rng.gauss(0, step_size))
    return f


def run(args):
    rng = random.Random(args.seed)

    start = (rng.uniform(-5, 5), rng.uniform(-5, 5))
    result = anneal(
        initial_state=start,
        energy=lambda s: rastrigin(*s),
        neighbor=gaussian_neighbor(args.step, rng),
        schedule=exponential_cooling(args.t0, args.alpha),
        n_iterations=args.iters,
        rng=rng,
    )

    print(f"start:      {start}, energy = {rastrigin(*start):.4f}")
    print(f"best found: {result.best_state}, energy = {result.best_energy:.4f}")
    print(f"global min: (0, 0), energy = 0.0000")
    print(f"accepted {sum(1 for s in result.history if s.accepted)} / {len(result.history)} proposals")

    animate(result, args)


def animate(result, args):
    history = result.history

    # background contour of the landscape
    grid = np.linspace(-5.12, 5.12, 300)
    X, Y = np.meshgrid(grid, grid)
    Z = np.vectorize(rastrigin)(X, Y)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[1, 1])

    ax_map.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax_map.set_title("Rastrigin landscape — yellow trail = accepted moves")
    ax_map.set_xlabel("x"); ax_map.set_ylabel("y")
    ax_map.plot(0, 0, marker="*", color="white", markersize=18,
                markeredgecolor="black", label="global min (0,0)")
    ax_map.legend(loc="upper right")

    trail_line, = ax_map.plot([], [], color="yellow", linewidth=1, alpha=0.7)
    current_pt, = ax_map.plot([], [], "rx", markersize=12, markeredgewidth=2)
    best_pt, = ax_map.plot([], [], marker="*", color="lime", markersize=15,
                           markeredgecolor="black", linestyle="None")

    iters = [s.iteration for s in history]
    cur_e = [s.current_energy for s in history]
    best_e = [s.best_energy_so_far for s in history]
    temps = [s.temperature for s in history]
    accept_p = [s.accept_probability if s.delta > 0 else float("nan")
                for s in history]

    ax_energy.set_xlim(0, len(history))
    ax_energy.set_ylim(0, max(cur_e) * 1.05)
    ax_energy.set_title("Energy over time")
    ax_energy.set_ylabel("energy")
    energy_line, = ax_energy.plot([], [], color="tab:blue",
                                  linewidth=0.8, label="current")
    best_line, = ax_energy.plot([], [], color="tab:red",
                                linewidth=1.2, linestyle="--", label="best-so-far")
    ax_energy.legend(loc="upper right")

    ax_temp.set_xlim(0, len(history))
    ax_temp.set_yscale("log")
    ax_temp.set_ylim(max(min(temps), 1e-4), max(temps) * 1.1)
    ax_temp.set_title("Temperature (log) + acceptance prob. of uphill moves")
    ax_temp.set_xlabel("iteration")
    temp_line, = ax_temp.plot([], [], color="tab:orange", label="T")
    ax_p = ax_temp.twinx()
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("P(accept worse)")
    p_scatter = ax_p.scatter([], [], s=4, color="tab:purple", alpha=0.4,
                             label="P(accept worse)")
    ax_temp.legend(loc="upper right")

    # we sample frames so long runs still animate at a sane rate
    n_frames = min(args.frames, len(history))
    frame_idx = np.linspace(0, len(history) - 1, n_frames).astype(int)

    accepted_path_x: list[float] = []
    accepted_path_y: list[float] = []
    last_drawn = -1

    def update(frame_i):
        nonlocal last_drawn
        upto = frame_idx[frame_i]
        # add every accepted step we skipped over to the trail
        for k in range(last_drawn + 1, upto + 1):
            step = history[k]
            if step.accepted:
                accepted_path_x.append(step.current_state[0]
                                       if k == 0 else step.proposed_state[0])
                accepted_path_y.append(step.current_state[1]
                                       if k == 0 else step.proposed_state[1])
        last_drawn = upto

        trail_line.set_data(accepted_path_x, accepted_path_y)
        cur_state = history[upto].current_state
        current_pt.set_data([cur_state[0]], [cur_state[1]])

        # best so far at this iteration
        best_e_here = history[upto].best_energy_so_far
        for s in history[:upto + 1]:
            if s.current_energy == best_e_here:
                best_pt.set_data([s.current_state[0]], [s.current_state[1]])
                break

        energy_line.set_data(iters[:upto + 1], cur_e[:upto + 1])
        best_line.set_data(iters[:upto + 1], best_e[:upto + 1])
        temp_line.set_data(iters[:upto + 1], temps[:upto + 1])

        pts = [(iters[i], accept_p[i]) for i in range(upto + 1)
               if not math.isnan(accept_p[i])]
        if pts:
            p_scatter.set_offsets(np.array(pts))

        ax_map.set_title(
            f"iter {upto}/{len(history) - 1}   "
            f"T = {history[upto].temperature:.3f}   "
            f"current E = {history[upto].current_energy:.3f}   "
            f"best E = {history[upto].best_energy_so_far:.4f}"
        )
        return trail_line, current_pt, best_pt, energy_line, best_line, temp_line

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
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--t0", type=float, default=5.0,
                   help="initial temperature (higher = more exploration)")
    p.add_argument("--alpha", type=float, default=0.997,
                   help="exponential cooling factor (closer to 1 = slower)")
    p.add_argument("--step", type=float, default=0.5,
                   help="neighbor step size (Gaussian std-dev)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=400,
                   help="animation frames (history is subsampled)")
    p.add_argument("--interval", type=int, default=25,
                   help="ms between frames")
    p.add_argument("--save", type=str, default=None,
                   help="optional path to save animation as .mp4 or .gif")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
