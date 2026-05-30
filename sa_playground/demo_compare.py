"""
Compare simulated annealing to two simpler baselines on the same problem.

This is the "why bother with SA?" demo. Three searches start from the
same random point on the Rastrigin landscape:

  * Random search         - sample N points uniformly, return the best.
  * Greedy hill descent   - always move to a better neighbor, stop when
                            none of the local neighbors are better.
  * Simulated annealing   - this is what we are teaching.

We run K trials of each (different starting points) and plot the
distribution of final energies. The takeaway you should see:

  * Random search is bad. It doesn't exploit the structure of the landscape.
  * Greedy is fast and usually finds *a* local minimum, but on Rastrigin
    almost never the global one — the histogram is wide.
  * SA finds the global minimum much more often. It is slower per trial
    than greedy, but the result quality is dramatically better.

Usage:
    python3 demo_compare.py
    python3 demo_compare.py --trials 30 --iters 3000
"""

from __future__ import annotations

import argparse
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from sa import anneal, exponential_cooling


def rastrigin(x: float, y: float) -> float:
    A = 10.0
    return (2 * A
            + (x * x - A * math.cos(2 * math.pi * x))
            + (y * y - A * math.cos(2 * math.pi * y)))


def random_search(rng, iters):
    best = (rng.uniform(-5, 5), rng.uniform(-5, 5))
    best_e = rastrigin(*best)
    for _ in range(iters):
        cand = (rng.uniform(-5, 5), rng.uniform(-5, 5))
        e = rastrigin(*cand)
        if e < best_e:
            best, best_e = cand, e
    return best_e


def greedy_descent(rng, iters, step=0.5):
    """Only accept improving neighbors. Stops at the first local minimum."""
    cur = (rng.uniform(-5, 5), rng.uniform(-5, 5))
    cur_e = rastrigin(*cur)
    for _ in range(iters):
        cand = (cur[0] + rng.gauss(0, step), cur[1] + rng.gauss(0, step))
        e = rastrigin(*cand)
        if e < cur_e:
            cur, cur_e = cand, e
    return cur_e


def sa_search(rng, iters, t0=5.0, alpha=0.997, step=0.5):
    start = (rng.uniform(-5, 5), rng.uniform(-5, 5))

    def neighbor(s):
        return (s[0] + rng.gauss(0, step), s[1] + rng.gauss(0, step))

    res = anneal(
        initial_state=start,
        energy=lambda s: rastrigin(*s),
        neighbor=neighbor,
        schedule=exponential_cooling(t0, alpha),
        n_iterations=iters,
        rng=rng,
        record_history=False,
    )
    return res.best_energy


def time_trials(name, fn, trials, seed_base):
    energies = []
    t0 = time.perf_counter()
    for k in range(trials):
        rng = random.Random(seed_base + k)
        energies.append(fn(rng))
    return name, np.array(energies), time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    results = [
        time_trials("random",  lambda r: random_search(r, args.iters),  args.trials, args.seed),
        time_trials("greedy",  lambda r: greedy_descent(r, args.iters), args.trials, args.seed),
        time_trials("anneal",  lambda r: sa_search(r, args.iters),       args.trials, args.seed),
    ]

    print(f"\nRastrigin minimization, {args.trials} trials, {args.iters} iters each")
    print(f"global minimum = 0.0\n")
    print(f"{'method':<10} {'mean':>10} {'median':>10} {'best':>10} {'worst':>10} {'time(s)':>10}")
    for name, e, t in results:
        print(f"{name:<10} {e.mean():>10.3f} {np.median(e):>10.3f} "
              f"{e.min():>10.3f} {e.max():>10.3f} {t:>10.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(r[1].max() for r in results) * 1.05, 30)
    for name, e, _ in results:
        ax.hist(e, bins=bins, alpha=0.55, label=name)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="global min")
    ax.set_xlabel("final energy (lower = better)")
    ax.set_ylabel("# trials")
    ax.set_title(f"Distribution of final energies over {args.trials} trials")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
