# Simulated Annealing playground

A self-contained sandbox for learning simulated annealing (SA), separate
from the main MATH 381 IP project. Nothing here imports from the parent
directory and the parent directory does not import from here.

## What is in here

| file              | what it does                                                  |
|-------------------|---------------------------------------------------------------|
| `sa.py`           | The core SA algorithm. Generic — you supply the problem.      |
| `demo_function.py`| Visual: SA on a bumpy 2D landscape (Rastrigin function).      |
| `demo_tsp.py`     | Visual: SA on the Traveling Salesman Problem.                 |
| `demo_compare.py` | Random vs. greedy vs. SA on the same problem, side by side.   |

## Quick start

You already have `matplotlib` and `numpy` installed via the parent project.
From the `sa_playground/` directory:

```bash
# the most pedagogical view — watch SA navigate a landscape with many traps
python3 demo_function.py

# combinatorial flavor (closer to your TA assignment problem)
python3 demo_tsp.py

# show why SA actually beats simpler methods
python3 demo_compare.py
```

## What you are looking at

Simulated annealing minimizes a function called the **energy**. Each
iteration:

1. Propose a small random change to the current state ("neighbor").
2. Compute `delta = energy(new) - energy(current)`.
3. If `delta <= 0` (new is better-or-equal), always accept.
4. If `delta > 0` (new is worse), accept with probability `exp(-delta / T)`.
5. Lower `T` slightly. Repeat.

The four knobs you actually tune:

| knob                | what it controls                                          | symptom of bad value                                       |
|---------------------|-----------------------------------------------------------|------------------------------------------------------------|
| initial temperature | how willing you are to accept bad moves at the start      | too low: stuck in starting basin. too high: random walk    |
| cooling schedule    | how fast `T` drops (we use `T_k = T0 * alpha^k`)          | too fast: traps. too slow: wasted compute                  |
| neighbor function   | how big a perturbation each proposal is                   | too small: cannot escape. too large: random search         |
| iterations          | total work budget                                         | not enough to cool properly                                |

Useful rules of thumb:
- Choose `T0` so that `exp(-typical_bad_delta / T0)` is roughly 0.5 to 0.9.
  That means roughly half of bad moves are accepted at the start.
- For `alpha`, pick something in `[0.99, 0.9995]` for most problems. The
  longer your run, the closer to 1.
- Iterations × cooling rate together determine final temperature. Aim for
  the final `T` to be tiny enough that nothing worse is being accepted
  anymore.

## Suggested experiments

In `demo_function.py`:

```bash
# default — should usually reach (0,0)
python3 demo_function.py

# cool way too fast — watch SA get trapped in a local minimum
python3 demo_function.py --alpha 0.95

# almost no temperature — equivalent to greedy descent
python3 demo_function.py --t0 0.01

# huge neighbor step — degenerates into random search
python3 demo_function.py --step 5.0

# slow cool with more iterations — should reliably find global min
python3 demo_function.py --iters 8000 --alpha 0.9995
```

In `demo_tsp.py`:

```bash
# bigger problem, more iters
python3 demo_tsp.py --cities 60 --iters 12000 --alpha 0.9992

# zero temperature — pure greedy 2-opt, watch it get stuck early
python3 demo_tsp.py --t0 1e-6
```

In `demo_compare.py`:

```bash
python3 demo_compare.py --trials 50 --iters 4000
```

You should see SA's histogram pile up near zero (the global minimum)
while greedy's histogram spreads across whatever local basins it
happened to land in.

## How this relates to the main project

The TA/course assignment problem in the parent directory is solved
exactly with Gurobi on small instances and with various heuristics on
large ones. Simulated annealing is another heuristic in that family —
it can handle the same kind of problem when:

- the instance is too large for exact IP,
- the objective is too messy to encode linearly,
- you want a "good-enough fast" answer rather than a proven optimum.

To port what you learn here to that problem, you would supply:
- **state** = a current assignment of TAs to courses,
- **energy** = `-objective + penalty * constraint_violations`,
- **neighbor** = small change like "move TA i off course c onto course c'",
- everything else is identical to what is in `sa.py`.

That extension is intentionally left out of this playground — the point
here is to build intuition for the algorithm first.
