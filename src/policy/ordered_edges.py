from enum import Enum
from typing import Dict, List, Tuple
from gmpy2 import mpq
from src.linpreds import Direction, DirectionSets
from src.backward_reachability import (
    BackwardReachabilityTree,
    BackwardReachabilityTreeNode,
    VertexInterval,
)
from src.polygons import HalfEdge, PolygonGridWorld, Vertex


class EdgeActions:
    NavigationDirection = Enum("NavigationDirection", ["Up", "Down", "Neutral"])

    def __init__(self, edge: HalfEdge) -> None:
        self.edge = edge
        self.targets: List[Tuple[VertexInterval, EdgeActions.NavigationDirection]] = []
        self.current_target = 0

    def add_target(self, vi: VertexInterval):
        if not self.targets:
            self.targets.append((vi.copy(), self.NavigationDirection.Neutral))
            return

        last_target, last_direction = self.targets.pop()
        if vi.start == last_target.end:
            self.targets.append(
                (VertexInterval(last_target.start, vi.end), self.NavigationDirection.Up)
            )
        elif vi.end == last_target.start:
            self.targets.append(
                (
                    VertexInterval(vi.start, last_target.end),
                    self.NavigationDirection.Down,
                )
            )
        else:
            self.targets.append((last_target, last_direction))
            self.targets.append((vi.copy(), self.NavigationDirection.Neutral))

    def restart(self) -> None:
        self.current_target = 0

    def navigate(self, coord: Vertex) -> HalfEdge:
        x_bounds = [coord.x, coord.x]
        y_bounds = [coord.y, coord.y]

        acts = DirectionSets[self.edge.actions] if self.edge.actions else ()
        for d in acts:
            if d == Direction.L:
                x_bounds[0] = None
            elif d == Direction.R:
                x_bounds[1] = None
            elif d == Direction.U:
                y_bounds[1] = None
            elif d == Direction.D:
                y_bounds[0] = None
            else:
                raise ValueError

        next_target, next_dir = None, None
        next_filter: List[VertexInterval] = []
        if self.current_target + 1 < len(self.targets):
            next_target, next_dir = self.targets[self.current_target + 1]

            next_x_filter = next_target.intersect_x_bounds(*x_bounds)
            for vi in next_x_filter:
                next_filter.extend(vi.intersect_y_bounds(*y_bounds))

        if next_filter:
            self.current_target += 1
            if next_dir == self.NavigationDirection.Neutral:
                return HalfEdge(
                    coord,
                    (next_filter[0].start + next_filter[0].end).mult_const(mpq(1, 2)),
                )
            elif next_dir == self.NavigationDirection.Up:
                return HalfEdge(coord, next_filter[0].end)
            elif next_dir == self.NavigationDirection.Down:
                return HalfEdge(coord, next_filter[0].start)
            else:
                raise ValueError

        target, dir = self.targets[self.current_target]
        x_filter = target.intersect_x_bounds(*x_bounds)
        filter: List[VertexInterval] = []
        for vi in x_filter:
            filter.extend(vi.intersect_y_bounds(*y_bounds))

        if not filter:
            raise ValueError

        if dir == self.NavigationDirection.Neutral:
            return HalfEdge(
                coord,
                (filter[0].start + filter[0].end).mult_const(mpq(1, 2)),
            )
        elif dir == self.NavigationDirection.Up:
            return HalfEdge(coord, filter[0].end)
        elif dir == self.NavigationDirection.Down:
            return HalfEdge(coord, filter[0].start)
        else:
            raise ValueError


class OrderedEdgePolicy:
    def __init__(
        self,
        gridw: PolygonGridWorld,
        btree: BackwardReachabilityTree,
        start_leaf: BackwardReachabilityTreeNode,
    ) -> None:
        self.gridw = gridw
        self.btree = btree
        self.start_leaf = start_leaf
        self.edge_actions: Dict[HalfEdge, EdgeActions] = dict()

    def build(self) -> None:
        curr = self.start_leaf
        while curr.forward_node and not curr.forward_node.contains_target:
            if not curr.linked_edge in self.edge_actions:
                self.edge_actions[curr.linked_edge] = EdgeActions(curr.linked_edge)

            self.edge_actions[curr.linked_edge].add_target(curr.forward_node.edge)

    def navigate(self, coord: Vertex, edge: HalfEdge) -> HalfEdge:
        assert edge in self.edge_actions
        return self.edge_actions[edge].navigate(coord)
