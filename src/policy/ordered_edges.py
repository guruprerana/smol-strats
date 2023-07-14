# obsolete file need to remove

from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple
from gmpy2 import mpq, to_binary, from_binary
from src.linpreds import Direction, DirectionSets
from src.backward_reachability import (
    BackwardReachabilityTree,
    BackwardReachabilityTreeNode,
    VertexInterval,
    VertexIntervalSerializer,
)
from src.polygons import (
    HalfEdge,
    PolygonGridWorld,
    PolygonGridWorldSerializer,
    Vertex,
    VertexSerializer,
)

NavigationDirection = Enum("NavigationDirection", ["Up", "Down", "Neutral"])


def serialize_nav_dir(nd: NavigationDirection) -> int:
    if nd == NavigationDirection.Neutral:
        return 0
    elif nd == NavigationDirection.Up:
        return 1
    elif nd == NavigationDirection.Down:
        return 2


def deserialize_nav_dir(nd: int) -> NavigationDirection:
    if nd == 0:
        return NavigationDirection.Neutral
    elif nd == 1:
        return NavigationDirection.Up
    elif nd == 2:
        return NavigationDirection.Down


class EdgeActions:
    def __init__(
        self,
        edge: HalfEdge,
        targets: Optional[
            List[Tuple[VertexInterval, NavigationDirection, HalfEdge]]
        ] = None,
    ) -> None:
        self.edge = edge
        self.targets: List[Tuple[VertexInterval, NavigationDirection, HalfEdge]] = (
            targets or []
        )
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

    def serialize(self) -> List[Tuple[str, str]]:
        return


class EdgeActionsSerializer:
    SerializedType = Tuple[
        int, List[Tuple[VertexIntervalSerializer.SerializedType, int, int]]
    ]

    def serialize(ea: EdgeActions, he_index: Dict[HalfEdge, int]) -> SerializedType:
        return he_index[ea.edge], [
            (
                VertexIntervalSerializer.serialize(vi),
                serialize_nav_dir(nd),
                he_index[he],
            )
            for (vi, nd, he) in ea.targets
        ]

    def deserialize(ser: SerializedType, he_index: List[HalfEdge]) -> EdgeActions:
        targets = [
            (
                VertexIntervalSerializer.deserialize(vi),
                deserialize_nav_dir(nd),
                he_index[he],
            )
            for (vi, nd, he) in ser[1]
        ]
        return EdgeActions(he_index[ser[0]], targets)


class OrderedEdgePolicy:
    def __init__(
        self,
        gridw: PolygonGridWorld,
        ea: Optional[Dict[HalfEdge, EdgeActions]] = None,
        start_edge: Optional[HalfEdge] = None,
        start_point: Optional[Vertex] = None,
        btree: Optional[BackwardReachabilityTree] = None,
        start_leaf: Optional[BackwardReachabilityTreeNode] = None,
    ) -> None:
        self.gridw = gridw
        self.built = True if ea else False
        self.edge_actions: Dict[HalfEdge, EdgeActions] = ea if ea else dict()
        self.start_edge: HalfEdge = start_edge or None

        if btree:
            assert btree.polygrid == gridw
        self.start_point = start_point if start_point else Vertex(0, 0)
        self.btree = btree

        if start_leaf:
            assert start_leaf in btree.all_leaves
        self.start_leaf = start_leaf

    def build(self, btree_depth=1000) -> None:
        if not self.btree:
            self.btree = BackwardReachabilityTree(self.gridw, self.start_point)
            self.btree.construct_tree(max_depth=btree_depth)

        if not self.start_leaf:
            self.start_leaf = self.btree.least_depth_begin_leaf()
            self.start_edge = self.start_leaf.linked_edge
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


class OrderedEdgePolicySerializer:
    SerializedType = Tuple[
        PolygonGridWorldSerializer.SerializedType,
        List[Tuple[int, EdgeActionsSerializer.SerializedType]],
        int,
    ]

    def serialize(p: OrderedEdgePolicy, start_point=Vertex(0, 0)) -> SerializedType:
        gridw_serializer = PolygonGridWorldSerializer(p.gridw)
        gridw_ser = gridw_serializer.serialize(start_point=start_point)
        eas = [
            (
                gridw_serializer.rev_he_index[he],
                EdgeActionsSerializer.serialize(ea, gridw_serializer.rev_he_index),
            )
            for (he, ea) in p.edge_actions.items()
        ]
        start_edge = gridw_serializer.rev_he_index[p.start_edge] if p.start_edge else -1

        return gridw_ser, eas, start_edge

    def deserialize(ser: SerializedType) -> OrderedEdgePolicy:
        gridw_serializer = PolygonGridWorldSerializer()
        gridw = gridw_serializer.deserialize(ser[0][:-1])
        start_point = VertexSerializer.deserialize(ser[0][-1])

        eas = {
            gridw_serializer.half_edges[he]: EdgeActionsSerializer.deserialize(
                ea, gridw_serializer.half_edges
            )
            for (he, ea) in ser[1]
        }

        start_edge = gridw_serializer.half_edges[ser[2]] if ser[2] != -1 else None

        return OrderedEdgePolicy(gridw, eas, start_edge, start_point=start_point)
