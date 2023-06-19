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

NavigationDirection = Enum("NavigationDirection", ["Up", "Down", "Neutral"])


class EdgeActions:
    def __init__(self, edge: HalfEdge) -> None:
        self.edge = edge
        self.targets: List[Tuple[VertexInterval, NavigationDirection, HalfEdge]] = []
        self.current_target = 0

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"EdgeActions({self.edge}, {self.targets}, {self.current_target})"

    def add_target(self, vi: VertexInterval, e: HalfEdge):
        if not self.targets:
            self.targets.append((vi.copy(), NavigationDirection.Neutral, e))
            return

        last_target, last_direction, last_e = self.targets.pop()
        if not e.ends_eq(last_e):
            self.targets.append((last_target, last_direction, last_e))
            self.targets.append((vi.copy(), NavigationDirection.Neutral, e))
            return

        if vi.start == last_target.end:
            self.targets.append(
                (
                    VertexInterval(last_target.start, vi.end),
                    NavigationDirection.Up,
                    e,
                )
            )
        elif vi.end == last_target.start:
            self.targets.append(
                (
                    VertexInterval(vi.start, last_target.end),
                    NavigationDirection.Down,
                    e,
                )
            )
        else:
            self.targets.append((last_target, last_direction, last_e))
            self.targets.append((vi.copy(), NavigationDirection.Neutral, e))

    def restart(self) -> None:
        self.current_target = 0

    def navigate(self, coord: Vertex) -> Tuple[HalfEdge, HalfEdge]:
        assert self.edge.contains_vertex(coord)
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

        next_target, next_dir, next_e = None, None, None
        next_filter: List[VertexInterval] = []
        if self.current_target + 1 < len(self.targets):
            next_target, next_dir, next_e = self.targets[self.current_target + 1]

            next_x_filter = next_target.intersect_x_bounds(x_bounds[0], x_bounds[1])
            for vi in next_x_filter:
                next_filter.extend(vi.intersect_y_bounds(y_bounds[0], y_bounds[1]))

        if next_filter:
            self.current_target += 1
            if next_dir == NavigationDirection.Neutral:
                return (
                    HalfEdge(
                        coord,
                        (next_filter[0].start + next_filter[0].end).mult_const(
                            mpq(1, 2)
                        ),
                    ),
                    next_e,
                )
            elif next_dir == NavigationDirection.Up:
                return HalfEdge(coord, next_filter[0].end), next_e
            elif next_dir == NavigationDirection.Down:
                return HalfEdge(coord, next_filter[0].start), next_e
            else:
                raise ValueError

        target, dir, e = self.targets[self.current_target]
        x_filter = target.intersect_x_bounds(x_bounds[0], x_bounds[1])
        filter: List[VertexInterval] = []
        for vi in x_filter:
            filter.extend(vi.intersect_y_bounds(y_bounds[0], y_bounds[1]))

        if not filter:
            raise ValueError

        if dir == NavigationDirection.Neutral:
            return (
                HalfEdge(
                    coord,
                    (filter[0].start + filter[0].end).mult_const(mpq(1, 2)),
                ),
                e,
            )
        elif dir == NavigationDirection.Up:
            return HalfEdge(coord, filter[0].end), e
        elif dir == NavigationDirection.Down:
            return HalfEdge(coord, filter[0].start), e
        else:
            raise ValueError


class OrderedEdgePolicy:
    def __init__(self, gridw: PolygonGridWorld) -> None:
        self.gridw = gridw
        self.built = False
        self.edge_actions: Dict[HalfEdge, EdgeActions] = dict()

    def build(self, btree_depth=100) -> None:
        self.btree = BackwardReachabilityTree(self.gridw)
        self.btree.construct_tree(max_depth=btree_depth)
        self.start_leaf = self.btree.least_depth_begin_leaf()
        assert self.start_leaf is not None

        curr = self.start_leaf
        while curr.forward_node and not curr.contains_target:
            if not curr.linked_edge in self.edge_actions:
                self.edge_actions[curr.linked_edge] = EdgeActions(curr.linked_edge)

            self.edge_actions[curr.linked_edge].add_target(
                curr.forward_node.edge, curr.forward_node.linked_edge
            )
            curr = curr.forward_node

        self.built = True

    def navigate(self, coord: Vertex, edge: HalfEdge) -> Tuple[HalfEdge, HalfEdge]:
        assert self.built
        assert edge in self.edge_actions
        return self.edge_actions[edge].navigate(coord)

    def restart(self) -> None:
        assert self.built
        for _, ea in self.edge_actions.items():
            ea.restart()
