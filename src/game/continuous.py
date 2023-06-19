from typing import Optional, Tuple
from src.linpreds import Direction, DirectionSets
from src.policy import BasePolicy
from src.polygons import HalfEdge, PolygonGridWorld, Vertex


class ContinuousReachabilityGridGame:
    def __init__(
        self, gridw: PolygonGridWorld, start_edge: HalfEdge, start_coord: Vertex
    ) -> None:
        self.gridw = gridw
        self.start_edge = start_edge
        self.current_edge = start_edge
        self.start_coord = start_coord
        self.current_coord = start_coord

        self.target_edges = set(gridw.target.edges_in_polygon())

    def restart(self) -> None:
        self.current_edge = self.start_edge
        self.current_coord = self.start_coord

    def _validate_direction(self, direction: HalfEdge) -> None:
        d = direction.end - direction.start
        acts = (
            DirectionSets[self.current_edge.actions]
            if self.current_edge.actions
            else ()
        )
        if d.y > 0:
            assert Direction.U in acts
        elif d.y < 0:
            assert Direction.D in acts

        if d.x > 0:
            assert Direction.R in acts
        elif d.x < 0:
            assert Direction.L in acts

    def step(self, direction: HalfEdge) -> Tuple[Vertex, Optional[HalfEdge], bool]:
        assert self.current_coord == direction.start
        self._validate_direction(direction)
        intersection_vertex = self.current_edge.intersects_edge(direction)
        intersection_edge = self.current_edge.next

        while not intersection_vertex and intersection_edge != self.current_edge:
            intersection_vertex = intersection_edge.intersects_edge(direction)
            intersection_edge = intersection_edge.next

        if not intersection_vertex:
            raise ValueError

        self.current_coord = intersection_vertex

        if not intersection_edge.opp:
            raise ValueError

        self.current_edge = intersection_edge.opp

        if self.current_edge in self.target_edges:
            return self.current_coord, self.current_edge, True

        return self.current_coord, self.current_edge, False

    def run(self, policy: BasePolicy, max_iter=1000) -> bool:
        target_reached = self.current_edge in self.target_edges
        for _ in range(max_iter):
            _, _, target_reached = self.step(
                policy.navigate(self.current_coord, self.current_edge)
            )
            if target_reached:
                break

        return target_reached
