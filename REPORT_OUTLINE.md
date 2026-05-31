# Report Outline — UW Math Scheduling Optimization

> **These are talking points, not final prose.** Your professor explicitly does
> not want AI-written text — rewrite every section in your own words. Use this to
> remember *what* to say, *which formulas* to show, and *which design choices*
> to justify. All numbers/parameters below were verified against the code on
> 2026-05-31.
>
> Sections follow the order required by the assignment:
> Abstract → Problem → Simplifications → Modeling → Solving → Data generation →
> Results → Improvements → Conclusion + References → Appendix.

---

## 0. The two problems we actually built (read this first)

The codebase solves **two nested versions** of the assignment problem. Keep them
straight throughout the report — every section maps onto one or both.

| | **Step 1** | **Step 2** |
|---|---|---|
| Question | who *teaches each class* | who teaches each class *and in which timeslot* |
| Decision unit | pair `(prof, course)` | triple `(prof, course, timeslot)` |
| Files | `solve.py` (IP), `solve_min_cost_flow.py` (flow) | `real_solve1.py` (IP), `real_solve1_sa.py` (SA) |
| Methods | Integer Program **and** Min-Cost Max-Flow | Integer Program **and** Simulated Annealing |

- We solve each step with **two independent methods** so we can *cross-check* and
  *compare* (exact vs. heuristic, general IP vs. specialized flow).
- "Professor/TA" are used interchangeably in the code (`TA` dataclass, `professor_id`
  columns) — say once in the report that an instructor is the assignable resource.

---

## 1. Abstract (≤ 10 lines)

- Problem: assign UW Math instructors to the department's autumn course sections
  (and, in the richer version, to timeslots) to maximize a preference/quality
  score subject to teaching-load, qualification, required-coverage, and
  no-double-booking constraints.
- Methods: exact **integer programming** (Gurobi); **min-cost max-flow** for the
  simpler prof→class subproblem; **simulated annealing** for the full
  prof→course→timeslot problem at scale.
- Headline result to state up front: on a realistic 54-prof / 59-course instance
  the IP is optimal **in 0.01 s**; SA reaches **99.3–99.9 %** of that optimum but
  takes ~10 s — so SA is only justified once the instance is too large for the IP.

---

## 2. Problem Description

- **Real-world motivation:** every term a math department must decide who teaches
  what. A good assignment respects expertise (don't put a topologist on a stats
  course), instructor workload, time-of-day preferences, room fit, and — above
  all — must *cover every required service course*. Doing this by hand is slow and
  political; an optimizer makes the trade-offs explicit.
- **Goal / objective:** produce a full, feasible teaching schedule that maximizes
  total "assignment quality" = preference + time-fit + course-importance + room-fit.
- **Why it's relevant:** it's a real operational problem at UW; it's also a clean
  instance of the **assignment / generalized-assignment** family from class.
- **Required UW Math courses modeled** (must be covered): MATH 111, 120, 124, 125,
  126, 207, 208, 300, 327, 381, 394, 402, 407. Everything else is *optional* (may
  be dropped if capacity is tight). Required-ness is detected from the course code
  in [data.py](data.py#L145) (`required_by_course_code`).
- **Inputs:** instructors (with max teaching load + research tags), course sections
  (with enrollment + topic tags + required flag), timeslots, rooms, and three
  preference layers (course, time, room-fit).

---

## 3. Simplifications (the professor specifically wants this section)

State each one *and why it was reasonable*:

- **Timeslots are pre-assigned per course (biggest simplification).** The data
  generator fixes each section's meeting time before solving, and marks every
  instructor available at every slot. Consequence we *measured*: the set of
  qualified `(prof,course)` pairs equals the set of qualified `(prof,course,timeslot)`
  triples (**679 = 679** in the default instance). So in the current data the
  "choose a timeslot" dimension collapses — Step 2 is solving over the same
  structure as Step 1, with the no-double-booking constraint as the only extra
  teeth. *Why OK:* models the common real case where the timetable grid is fixed
  and you're staffing it. *Flag it honestly* as the thing most worth removing.
- **Instructor time-availability is never binding.** Availability only shifts the
  *score* (`time_weight`), never feasibility, because everyone is "available"
  everywhere in the generated data.
- **Rooms are not a decision.** The solver never assigns rooms; room quality enters
  as a static per-course number = score of the *best* room that fits
  (`best_room_score`). So we ignore room *contention* (two classes wanting the same
  room at the same time). *Why OK:* keeps the model a pure instructor-assignment
  problem; room contention would couple courses and make it a much harder timetabling problem.
- **Demand is ~1 instructor per course.** `STUDENTS_PER_TA = 1000` in
  [data.py](data.py#L15) ⇒ `min_tas[c] = ceil(enrollment/1000) = 1` for every
  course. So "demand" is trivial in the default instance; capacity rarely binds.
  Mention you can lower this constant (e.g. 30) to stress-test the demand constraint.
- **Preferences are synthetic**, derived from research-tag overlap, not real stated
  preferences (see §6). Ranks are bucketed into a few fixed weights rather than
  continuous utilities.
- **One department, one term**; no prerequisites/sequencing, no cross-listing, no
  multi-section coordination, no student-level demand beyond an enrollment count.
- **The objective is a weighted *sum* of heterogeneous terms** (preference + time +
  priority + room). This assumes the terms are commensurable and that the
  hand-chosen weights encode the right priorities (see §4 for the actual numbers).

---

## 4. Modeling the Problem

> Tie this to class: Step 1 is the **assignment / transportation problem**; the
> flow version is a **min-cost max-flow** on a bipartite network; Step 2 is a
> **0/1 integer program** (generalized assignment + conflict constraints).

### 4a. Common preprocessing design choice — *qualification by omission*

- We **only create a variable for qualified pairs/triples.** Unqualified
  combinations simply don't exist in the model, instead of being added as
  `x = 0` constraints. This shrinks the model and removes a whole constraint class.
- Built in [data.py](data.py#L253) (`qualified_pairs`) and
  [data.py](data.py#L283) (`qualified_triples`).

### 4b. Step 1 IP — prof → class ([solve.py](solve.py))

- Variable: `x[i,c] = 1` if instructor `i` teaches class `c`, for `(i,c) ∈ Q` (qualified).
- Objective: `maximize Σ_{(i,c)∈Q} w[i,c] · x[i,c]`
- Constraints:
  - `Σ_i x[i,c] ≥ d[c]` for every course `c` (meet demand)
  - `Σ_c x[i,c] ≤ L[i]` for every instructor `i` (max load)
- `d[c] = min_tas[c]` (=1 here), `L[i] = max_load`.
- **Design choice:** demand is `≥ d[c]`, *not* `= d[c]`. Because all weights are
  positive the IP may pile extra instructors onto a course if capacity is free.
  Note this and say what `=` would change.
- Weights `w[i,c]` (`PREF_WEIGHTS`, [solution.py](solution.py#L19)):

  | rank | 1 | 2 | 3 | 4 | none |
  |---|---|---|---|---|---|
  | weight | 0.35 | 0.25 | 0.15 | 0.10 | 0.05 |

### 4c. Step 1 as Min-Cost Max-Flow ([solve_min_cost_flow.py](solve_min_cost_flow.py))

- Network: `source → instructors → classes → sink`.
- Edges (capacity, cost):

  | edge | capacity | cost |
  |---|---|---|
  | `source → i` | `L[i]` (max load) | 0 |
  | `i → c` (qualified) | 1 | `−round(w[i,c]·1000)` |
  | `c → sink` | `d[c]` (demand) | 0 |

- Flow `f[i,c]=1` ⇔ instructor `i` assigned to class `c`.
- We push exactly `target_flow = Σ_c d[c]` units. **Design choice / difference from
  the IP:** the flow assigns *exactly* `d[c]` per class (the IP allows ≥).
- **Why negative, scaled costs:** min-cost flow *minimizes*, so we negate weights to
  turn it into preference *maximization*; we scale by `COST_SCALE = 1000`
  ([solve_min_cost_flow.py](solve_min_cost_flow.py#L30)) so costs are integers.
- Feasibility shortcuts: if `total_capacity < total_demand` → INFEASIBLE before
  solving; if the final flow `< total_demand` → INFEASIBLE.
- *Correction to your notes:* this is **fully implemented** (252 lines, working),
  not a TODO.

### 4d. Step 2 IP — prof → course → timeslot ([real_solve1.py](real_solve1.py))

- Variable: `x[i,c,t] = 1` if instructor `i` teaches course `c` at timeslot `t`,
  for `(i,c,t) ∈ qualified_triples`.
- Objective: `maximize Σ score[i,c,t] · x[i,c,t]` where
  `score = α[i]·course_pref_weight + β[i]·time_weight + course_priority_weight + room_weight`
  ([real_solve1.py](real_solve1.py#L79)).
- Constraints:
  - `Σ_{i,t} x[i,c,t] = 1` for each **required** course (cover exactly once)
  - `Σ_{i,t} x[i,c,t] ≤ 1` for each **optional** course (may drop)
  - `Σ_{c,t} x[i,c,t] ≤ L[i]` (max load)
  - `Σ_c x[i,c,t] ≤ 1` for each `(i,t)` (**no double-booking** a timeslot)
- Professor preference mode from `professors.csv`:

  | priority mode | `α[i]` course multiplier | `β[i]` time multiplier | meaning |
  |---|---:|---:|---|
  | `interest_first` | 1.50 | 0.75 | professor cares more about course/topic fit |
  | `time_first` | 0.75 | 1.50 | professor cares more about teaching time |
  | `balanced` / missing | 1.00 | 1.00 | fallback/default behavior |

- Design choice: these multipliers only affect personal preferences. Required-course
  priority and room suitability are department/course properties, so they are not
  changed by the professor's preference mode.
- The four base weight tables (this is the "what I chose for weights" content):

  | course-pref rank | 1 | 2 | 3 | 4 | none |
  |---|---|---|---|---|---|
  | weight | 0.35 | 0.25 | 0.15 | 0.10 | 0.05 |

  | time score | 5 | 4 | 3 | 2 | 1 |
  |---|---|---|---|---|---|
  | weight | 0.40 | 0.30 | 0.20 | 0.10 | 0.05 |

  | course priority | required | optional |
  |---|---|---|
  | weight | **2.50** | 0.25 |

  | room score | 5 | 4 | 3 | 2 | 1 |
  |---|---|---|---|---|---|
  | weight | 1.00 | 0.85 | 0.70 | 0.25 | 0.15 |

- **Design choices to defend:**
  - The big `2.50` on *required* dwarfs every other term — this is what makes the
    optimizer prioritize covering required service courses over chasing nice-to-haves.
  - Room is a *score*, not a variable (see §3).
  - Required `= 1` (exactly once) vs optional `≤ 1` (droppable) encodes the
    real distinction between service courses and electives.

### 4e. Worked mini-example (include a tiny figure)

- 3 instructors, 2 required + 1 optional course, 2 timeslots. Draw: the flow
  network for Step 1, and a small `x[i,c,t]` table for Step 2 with the chosen cells
  highlighted. This makes the formulation concrete for the reader.

---

## 5. Solving the Problem

### 5a. Exact IP — Gurobi
- Library: **Gurobi** (`gurobipy ≥ 11`, we ran 13.0.2). Branch-and-bound over the
  LP relaxation; returns a certificate of optimality (status OPTIMAL) and, on
  timeout, an optimality *gap*.
- ~20 lines of modeling each; the heavy lifting is the solver.

### 5b. Min-cost max-flow — implemented from scratch
- **Successive Shortest Paths** with **Johnson potentials** + **Dijkstra**
  (binary heap), using *reduced costs* so all edge costs stay non-negative after
  the first relabel. Custom `Edge`/`MinCostMaxFlow` classes with residual (reverse)
  edges ([solve_min_cost_flow.py](solve_min_cost_flow.py#L42)).
- Nice optimization to mention: initial potentials are seeded from each course's
  best incoming edge cost, avoiding a Bellman-Ford warm-up pass.
- This is "reimplement/adapt an algorithm from class" — good to highlight.

### 5c. Simulated Annealing — the main heuristic ([real_solve1_sa.py](real_solve1_sa.py))

Show pseudo-code, then walk the six design decisions.

```
s ← random_start(); T ← T0
best ← s
for level in 1..L:
    repeat M times:
        s' ← propose_neighbor(s)
        Δ ← E(s',T) − E(s,T)
        if Δ ≤ 0 or random() < exp(−Δ/T):   # Metropolis
            s ← s'
            if feasible(s) and obj(s) > obj(best): best ← s
    T ← α·T                                   # geometric cooling
return best   # best FEASIBLE schedule seen
```

- **(0) State:** `frozenset` of `(prof,course,timeslot)` triples — one proposed schedule.
- **(1) Random start** ([line 110](real_solve1_sa.py#L110)): assign required courses
  only; **shuffle the required-course order and each course's candidate list** to
  kill CSV/scrape ordering bias; greedily avoid overload/double-booking, else take
  any qualified triple and let the penalty fix it later. *Design point:* start
  near-feasible but don't demand feasibility.
- **(2) Energy / cost** ([line 166](real_solve1_sa.py#L166)):

  ```
  E(s,T) = −objective(s) + (100 / T) · violations(s)
  violations(s) = missing_required(s) + double_bookings(s)
  ```

  - Negate the objective because SA *minimizes*.
  - `100/T` makes constraint violations *increasingly* expensive as it cools —
    early on it can wander through infeasible states, late on it's forced feasible.
  - **Key design insight (put this in the report):** the violation summary actually
    tracks **6** types — `invalid_triples, missing_required, extra_required,
    overcovered_optional, overloads, double_bookings`
    ([line 683](real_solve1_sa.py#L683)) — but the cost penalizes only **2**. That's
    deliberate: the move set *can only ever create those two*. The other four are
    prevented *by construction* (moves use only qualified triples, check
    `can_add_without_overload`, and only add a course when its count is 0). So
    penalizing 2 is sufficient *for these moves* — a clean coupling between the
    neighbor design and the penalty design.
- **(3) Acceptance** ([line 602](real_solve1_sa.py#L602)): Metropolis
  `p = min(1, exp(−Δ/T))`. Δ≤0 always accepted; uphill moves accepted with prob that
  shrinks as T falls.
- **(4) Starting temperature** ([line 181](real_solve1_sa.py#L181)): sample many
  "typical bad moves," take the median uphill Δ, and solve
  `target = exp(−Δ/T0)` ⇒ **`T0 = −Δ_typical / ln(target)`**. Targets:

  | profile | target uphill-accept |
  |---|---|
  | low | 30 % |
  | medium | 60 % |
  | high | 80 % |

  (sample size = `max(100, min(1000, 2·|triples|))`; the probe uses a reference cost
  `−obj + total_violations` at T=1, not the real cost, to avoid circularity.)
- **(5) States per temperature** ([line 221](real_solve1_sa.py#L221)):
  `M = max(1, ⌊10·√|qualified_triples|⌋)`. **Design rationale:** bigger instances
  deserve more neighbors per level, but √ grows gently so it doesn't explode.
  - `10·√10,000 = 1,000`; `10·√1,000,000 = 10,000`; here `10·√679 ≈ 260`.
- **(6) Cooling** ([line 227](real_solve1_sa.py#L227)): geometric `T ← 0.98·T`.
  0.95 = fast/greedy, 0.99–0.999 = slow/thorough; 0.98 is the middle. Interacts with
  `levels` (default 200) and `M`.
- **(7) Neighbor moves** ([line 237](real_solve1_sa.py#L237)) — the move set:
  `change_professor`, `change_timeslot`, `replace_assignment`, `add_optional`,
  `remove_optional`, `remove_required`, `add_required`. (`swap_required_timeslots`
  is written but **commented out** — mention it as a disabled move.) Some moves
  *intentionally* allow infeasibility (`remove_required`) so SA can tunnel out of
  local minima and let the penalty pull it back.
- **What SA returns:** the **best *feasible* state** seen (feasible = all 6
  violations 0). Report distinguishes three things: returned best-feasible vs.
  final-current state vs. best-cost state — they can differ.

---

## 6. Data Generation ([generate_uw_dataset.py](generate_uw_dataset.py))

- **Base, real-derived:** scraped UW Math **Autumn 2026** schedule → **59 fixed
  lecture sections** (number, title, credits, enrollment, topic tags) and **104
  named UW Math faculty/instructors**, each
  hand-tagged with research interests + a max load. *This directly does your two
  finished TODOs: scrape faculty interests, scrape the class schedule.*
- **Per-seed randomized but reproducible** (`random.Random(seed)`):
  - **Timeslot per section** — weighted sampling (`pick_timeslot`,
    [line 295](generate_uw_dataset.py#L295)) encoding your schedule assumptions:
    5-credit lower-division → MWF daytime; 4-credit → MWF/TTh; ≤3-credit/grad →
    MW/TTh/seminar; seminars → late afternoon; big classes pushed out of late slots;
    nothing before 8:30, nothing regular after 5:20 except seminars.
  - **Room per section** — random among the 4 smallest rooms that still fit enrollment.
  - **Time preferences** — each instructor gets a profile
    (morning/midday/afternoon/flexible) → a 1–5 `time_pref_score` per slot, with jitter.
  - **Course preferences from tag overlap** (`topic_score`,
    [line 333](generate_uw_dataset.py#L333)): `min(5, 2 + |prof_tags ∩ course_tags|)`,
    with fallbacks (general_math/education/calculus). `qualified ⇔ score > 0`; rank
    bucketed `5→1, 4→2, 3→3, 2→4, else none`. **This is the "professor who does
    optimization research wants to teach optimization" idea, implemented.**
  - **Room suitability:** `suitable ⇔ capacity ≥ enrollment`; score 5/4/3 by slack;
    −1 if a big (≥80) class lacks a projector.
- **Reproducibility:** fixed seeds ⇒ identical files. Default = `seed_381`; we also
  generated seeds 1, 382–387 for variance testing, plus a stress instance.
- **Stress generator** (`--stress --scale 25`): clone every section ×25 → 1475
  sections, and add broad-tag "Stress Lecturer Pool" instructors (max load 12) →
  254 instructors. This is the *intentionally-too-big* instance.
- Output: one folder of CSVs per scenario (`professors`, `courses`, `timeslots`,
  `rooms`, `professor_course_preferences`, `professor_time_preferences`,
  `room_suitability`, `allowed_assignments`).

### Measured instance sizes (use these exact numbers)

| | default `seed_381` | stress `scale_25` |
|---|---|---|
| instructors | 54 | 254 |
| courses (req / opt) | 59 (29 / 30) | 1475 (725 / 750) |
| timeslots / rooms | 18 / 15 | 18 / 15 |
| qualified pairs = triples | 679 | 311,975 |
| total demand / capacity | 59 / 116 | 1475 / 2516 |

---

## 7. Results

### 7a. Small instance — IP vs SA (seed 381, SA seed 7, 200 levels). **Real, measured:**

| method | objective | % of IP | status | time |
|---|---|---|---|---|
| **IP (exact)** | **167.55** | 100 % | OPTIMAL | **0.010 s** |
| SA low (30 %) | 167.35 | 99.88 % | feasible | 9.5 s |
| SA medium (60 %) | 167.35 | 99.88 % | feasible | 9.5 s |
| SA high (80 %) | 166.40 | 99.31 % | feasible | 9.9 s |

- SA does **52,000 proposals per profile** (200 levels × 260 moves), all 59 courses
  covered, 0 violations.
- **Interpretation (and an honest, slightly counterintuitive finding):** on a small,
  near-feasible instance the *lower* starting temperature did best — high T wasted
  budget exploring. And the IP is both exact *and* ~1000× faster. So **SA is not
  worth it here**; its only justification is instances the IP can't handle.
- Acceptance rates rose with temperature (48.8 % → 55.5 % → 62.0 %) as expected —
  good sanity check that the T0 calibration works.

### 7b. When does the IP actually fail? (your open TODO — now tested, with real numbers)

- Your note said "IP fails because it's too big." **Tested — and the failure is
  sharper and more interesting than "too big/slow":**
  - The 311,975-variable stress IP does **not** time out. `m.optimize()` raises
    **`GurobiError: Model too large for size-limited license`** — it never reaches
    branch-and-bound at all.
  - We then binary-searched the cap directly: this (free/restricted) Gurobi license
    solves **exactly up to 2,000 variables**, and errors above it.
- **So the concrete input-size limit for the IP path is ≈ 2,000 qualified triples:**

  | instance | qualified triples | IP path |
  |---|---|---|
  | default `seed_381` | 679 | ✅ solves (0.01 s) |
  | 2× clone | 1,358 | ✅ solves |
  | 3× clone | 2,037 | ❌ license error |
  | stress `scale_25` | 311,975 | ❌ license error |

  i.e. the IP cuts out between scale 2 and 3 of the base instance.
- **Honest framing for the report:** the wall we hit is a *tooling/licensing* limit
  (the size-limited Gurobi license), not proof that the algorithm itself is
  intractable — the assignment-like structure (required `=1`, optional `≤1`, load
  `≤`, conflict `≤1`) often has a near-integral LP relaxation, so a full academic
  license might solve far larger. **But within this project's setup, beyond ~2,000
  variables SA is the only method that runs — which is exactly what motivates it.**
- Other genuine failure modes to mention: **infeasibility** (a required course with
  no qualified instructor; total capacity < total demand; timeslot conflicts that
  make required coverage impossible). The flow solver detects the capacity case
  up-front and returns INFEASIBLE without solving.

### 7c. Large instance — SA (from your notes; rerun and record cleanly)

- 1 min: feasible, objective ≈ **2912.55**.
- 5 min: **penalty not aggressive enough** — of ~200,000 states searched only ~2
  were feasible. *This is a real, reportable weakness*, not a failure to hide.
- Interpretation: at this scale the `100/T` penalty is too weak relative to the
  725-required-course coverage problem; SA spends almost all its time infeasible and
  the returned schedule is whatever rare feasible state it stumbled on.

### 7d. Robustness tests to run (don't ship one-instance results)

- **Variance:** run SA across seeds 1, 381–387 and report mean ± spread of the gap.
- **SA seed variance:** same instance, different `--seed`, to show run-to-run spread.
- **Input-size sweep:** small → `scale 5, 10, 25`; plot IP time and SA gap vs size to
  find where (if anywhere) the IP becomes impractical.
- **Baseline to beat:** a greedy "assign each required course to its highest-pref
  available instructor" — show SA/IP beat it.

---

## 8. Improvements / Future Work

- **Fix the large-instance SA penalty** (the concrete observed bug): cap or rescale
  the `100/T` term, or use an *adaptive* penalty that grows when feasibility is rare;
  add **repair moves** (fix a missing required course / a double-booking directly)
  instead of relying on random tunnelling.
- **Adaptive move probabilities** (your own idea): bias toward the move that reduces
  current violations instead of uniform `rng.choice` — should reach feasibility faster.
- **Remove the biggest simplification:** let the model *choose* timeslots (don't
  pre-fix them) and make instructors genuinely unavailable at some slots — this is
  where Step 2 becomes strictly richer than Step 1 and the SA earns its place.
- **Real room assignment** as a decision with contention (no two classes in one room
  at one time) — turns it into a true timetabling problem.
- **Exact-demand variant** (`= d[c]`) and a min-load floor; lower `STUDENTS_PER_TA`
  so demand actually binds.
- **Systematic comparison:** objective-vs-time curves, feasibility-rate-vs-time for SA.

---

## 9. Conclusion + References

- **Conclusion:** IP is the method of choice whenever it runs (exact + fast here);
  min-cost flow is the natural, dependency-free tool for the pure prof→class
  subproblem; SA is the scalable fallback for the full scheduling problem but needs
  careful penalty/move design and gives no optimality certificate. Be honest that
  on the instances we could fully solve, the IP won — SA's value is conditional on
  scale and on the model gaining harder constraints.
- **References:** Gurobi optimizer + `gurobipy` docs; a min-cost max-flow / SSP +
  potentials reference (Ahuja–Magnanti–Orlin *Network Flows*, or course notes); a
  simulated annealing reference (Kirkpatrick et al. 1983); UW Time Schedule
  (Autumn 2026 Math) as the data source; Python `matplotlib`/`numpy`.

---

## 10. Appendix / Extras

- Full weight tables (above), full move-type list.
- The SA animation: `python3 real_solve1_sa.py --scenario uw_math_autumn_2026_seed_381`
  produces an objective/violation/temperature/acceptance plot — screenshot it
  (assignment matrix + the four traces) instead of pasting code.
- Commands to regenerate everything (data gen + all three solvers).
- Small code excerpts only (the cost function and one neighbor move are the most
  worth showing) — keep them short and commented per the assignment's warning.

---

### Figures/tables checklist
- [ ] required-courses table · [ ] all four weight tables · [ ] flow-network diagram
- [ ] worked mini-example · [ ] instance-size table (§6) · [ ] IP-vs-SA results table (§7a)
- [ ] SA traces screenshot · [ ] size-sweep plot (IP time & SA gap vs scale)
