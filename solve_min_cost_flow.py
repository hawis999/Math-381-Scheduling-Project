"""Step-1 professor-to-class assignment via min-cost max-flow.

This solves the same smaller assignment problem as solve.py:

    source -> professor -> class -> sink

Edges:
  * source -> professor: capacity = professor max_load, cost = 0
  * professor -> class: capacity = 1, cost = -preference_score
  * class -> sink: capacity = class demand, cost = 0

Min-cost flow minimizes cost, so preference scores are negated. Sending the
maximum required flow gives the highest-preference feasible assignment.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from data import ScenarioData
from solution import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    Solution,
    preference_weight,
)

COST_SCALE = 1000


@dataclass
class Edge:
    to: int
    rev: int
    capacity: int
    cost: int
    initial_capacity: int


class MinCostMaxFlow:
    def __init__(self, node_count: int) -> None:
        self.graph: list[list[Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, src: int, dst: int, capacity: int, cost: int) -> Edge:
        forward = Edge(
            to=dst,
            rev=len(self.graph[dst]),
            capacity=capacity,
            cost=cost,
            initial_capacity=capacity,
        )
        backward = Edge(
            to=src,
            rev=len(self.graph[src]),
            capacity=0,
            cost=-cost,
            initial_capacity=0,
        )
        self.graph[src].append(forward)
        self.graph[dst].append(backward)
        return forward

    def min_cost_flow(
        self,
        source: int,
        sink: int,
        target_flow: int,
        potentials: list[int] | None = None,
    ) -> tuple[int, int]:
        """Return (flow_sent, total_scaled_cost)."""
        node_count = len(self.graph)
        potentials = potentials or [0] * node_count
        flow = 0
        total_cost = 0

        while flow < target_flow:
            distances = [float("inf")] * node_count
            previous_node = [-1] * node_count
            previous_edge_index = [-1] * node_count
            distances[source] = 0
            heap: list[tuple[float, int]] = [(0, source)]

            while heap:
                distance, node = heapq.heappop(heap)
                if distance != distances[node]:
                    continue

                for edge_index, edge in enumerate(self.graph[node]):
                    if edge.capacity <= 0:
                        continue
                    reduced_cost = edge.cost + potentials[node] - potentials[edge.to]
                    next_distance = distance + reduced_cost
                    if next_distance < distances[edge.to]:
                        distances[edge.to] = next_distance
                        previous_node[edge.to] = node
                        previous_edge_index[edge.to] = edge_index
                        heapq.heappush(heap, (next_distance, edge.to))

            if distances[sink] == float("inf"):
                break

            for node, distance in enumerate(distances):
                if distance < float("inf"):
                    potentials[node] += int(distance)

            pushed = target_flow - flow
            node = sink
            while node != source:
                prev = previous_node[node]
                edge = self.graph[prev][previous_edge_index[node]]
                pushed = min(pushed, edge.capacity)
                node = prev

            node = sink
            while node != source:
                prev = previous_node[node]
                edge_index = previous_edge_index[node]
                edge = self.graph[prev][edge_index]
                edge.capacity -= pushed
                self.graph[node][edge.rev].capacity += pushed
                total_cost += pushed * edge.cost
                node = prev

            flow += pushed

        return flow, total_cost


def solve(data: ScenarioData) -> Solution:
    """Solve step 1 using min-cost max-flow."""
    start = time.perf_counter()

    professors = sorted(data.tas, key=lambda ta: ta.ta_id)
    courses = sorted(data.courses, key=lambda course: course.course_id)
    total_demand = sum(data.min_tas.get(course.course_id, 0) for course in courses)

    if total_demand == 0:
        return Solution(
            status=STATUS_OPTIMAL,
            objective=0.0,
            selected_pairs=[],
            solver_time_sec=time.perf_counter() - start,
            notes="min-cost max-flow solved empty-demand instance",
        )

    total_capacity = sum(max(0, professor.max_load) for professor in professors)
    if total_capacity < total_demand:
        return Solution(
            status=STATUS_INFEASIBLE,
            objective=None,
            selected_pairs=[],
            solver_time_sec=time.perf_counter() - start,
            notes=(
                f"total professor capacity {total_capacity} is less than "
                f"total course demand {total_demand}"
            ),
        )

    source = 0
    professor_offset = 1
    course_offset = professor_offset + len(professors)
    sink = course_offset + len(courses)
    node_count = sink + 1

    professor_node = {
        professor.ta_id: professor_offset + idx
        for idx, professor in enumerate(professors)
    }
    course_node = {
        course.course_id: course_offset + idx
        for idx, course in enumerate(courses)
    }

    mcmf = MinCostMaxFlow(node_count)
    pair_edges: dict[tuple[str, str], Edge] = {}
    initial_potentials = [0] * node_count

    for professor in professors:
        if professor.max_load > 0:
            mcmf.add_edge(source, professor_node[professor.ta_id], professor.max_load, 0)

    best_incoming_course_cost: dict[str, int] = {}
    for professor_id, course_id in sorted(data.qualified_pairs):
        if professor_id not in professor_node or course_id not in course_node:
            continue

        pref = data.pref_by_pair.get((professor_id, course_id))
        if pref is None:
            continue

        scaled_score = int(round(preference_weight(pref) * COST_SCALE))
        cost = -scaled_score
        edge = mcmf.add_edge(
            professor_node[professor_id],
            course_node[course_id],
            capacity=1,
            cost=cost,
        )
        pair_edges[(professor_id, course_id)] = edge
        best_incoming_course_cost[course_id] = min(
            best_incoming_course_cost.get(course_id, cost),
            cost,
        )

    for course in courses:
        demand = data.min_tas.get(course.course_id, 0)
        if demand > 0:
            mcmf.add_edge(course_node[course.course_id], sink, demand, 0)
        initial_potentials[course_node[course.course_id]] = best_incoming_course_cost.get(
            course.course_id,
            0,
        )

    initial_potentials[sink] = min(
        (initial_potentials[course_node[course.course_id]] for course in courses),
        default=0,
    )

    flow_sent, _scaled_cost = mcmf.min_cost_flow(
        source=source,
        sink=sink,
        target_flow=total_demand,
        potentials=initial_potentials,
    )

    selected_pairs = [
        pair
        for pair, edge in pair_edges.items()
        if edge.initial_capacity - edge.capacity > 0
    ]
    selected_pairs.sort(key=lambda pair: (pair[1], pair[0]))

    elapsed = time.perf_counter() - start
    if flow_sent < total_demand:
        return Solution(
            status=STATUS_INFEASIBLE,
            objective=None,
            selected_pairs=selected_pairs,
            solver_time_sec=elapsed,
            notes=f"only sent {flow_sent}/{total_demand} required units of flow",
        )

    objective = sum(preference_weight(data.pref_by_pair[pair]) for pair in selected_pairs)
    return Solution(
        status=STATUS_OPTIMAL,
        objective=objective,
        selected_pairs=selected_pairs,
        solver_time_sec=elapsed,
        notes=f"min-cost max-flow sent {flow_sent}/{total_demand} required units of flow",
    )
