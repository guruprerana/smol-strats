from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple
from gmpy2 import mpq, to_binary, from_binary
from src.linpreds import Direction, DirectionSets
from src.backward_reachability import (
    BackwardReachabilityTree,
    BackwardReachabilityTreeNode,
    VertexInterval,
    VertexIntervalSerializer,
    leq,
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


def str_nav_dir(nd: NavigationDirection) -> str:
    if nd == NavigationDirection.Neutral:
        return "Neutral"
    elif nd == NavigationDirection.Up:
        return "Up"
    elif nd == NavigationDirection.Down:
        return "Down"


class SubgoalEdgeActions:
    def __init__(
        self,
        edge: HalfEdge,
        targets: Optional[
            Dict[
                VertexInterval,
                List[Tuple[VertexInterval, NavigationDirection, HalfEdge]],
            ]
        ] = None,
    ) -> None:
        self.edge = edge
        self.targets: Dict[
            VertexInterval, List[Tuple[VertexInterval, NavigationDirection, HalfEdge]]
        ] = (targets or dict())
        self.current_targets: Dict[VertexInterval, int] = dict()
        if targets:
            for subgoal in targets:
                self.current_targets[subgoal] = 0

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"EdgeActions({self.edge}, {self.targets})"

    def add_target(self, vi: VertexInterval, e: HalfEdge, subgoal: VertexInterval):
        if subgoal not in self.targets:
            self.targets[subgoal] = []
        if subgoal not in self.current_targets:
            self.current_targets[subgoal] = 0

        if not self.targets[subgoal]:
            self.targets[subgoal].append((vi.copy(), NavigationDirection.Neutral, e))
            return

        last_target, last_direction, last_e = self.targets[subgoal].pop()
        if not e.ends_eq(last_e) or e == self.edge:
            # we only want to merge vertex intervals when you go to different edge
            self.targets[subgoal].append((last_target, last_direction, last_e))
            self.targets[subgoal].append((vi.copy(), NavigationDirection.Neutral, e))
            return

        if leq(last_target.end, vi.start):
            self.targets[subgoal].append(
                (
                    VertexInterval(last_target.start, vi.end),
                    NavigationDirection.Up,
                    e,
                )
            )
        elif leq(vi.end, last_target.start):
            self.targets[subgoal].append(
                (
                    VertexInterval(vi.start, last_target.end),
                    NavigationDirection.Down,
                    e,
                )
            )
        else:
            self.targets[subgoal].append((last_target, last_direction, last_e))
            self.targets[subgoal].append((vi.copy(), NavigationDirection.Neutral, e))

    def restart(self) -> None:
        for vi in self.current_targets.keys():
            self.current_targets[vi] = 0

    def navigate(
        self, coord: Vertex, subgoal: VertexInterval
    ) -> Tuple[HalfEdge, HalfEdge]:
        assert self.edge.contains_vertex(coord)
        assert subgoal in self.targets
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
        if self.current_targets[subgoal] + 1 < len(self.targets[subgoal]):
            next_target, next_dir, next_e = self.targets[subgoal][
                self.current_targets[subgoal] + 1
            ]

            next_x_filter = next_target.intersect_x_bounds(x_bounds[0], x_bounds[1])
            for vi in next_x_filter:
                next_filter.extend(vi.intersect_y_bounds(y_bounds[0], y_bounds[1]))

        if next_filter:
            self.current_targets[subgoal] += 1
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

        target, dir, e = self.targets[subgoal][self.current_targets[subgoal]]
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


class SubgoalEdgeActionsSerializer:
    SerializedType = Tuple[
        int,
        List[
            Tuple[
                VertexIntervalSerializer.SerializedType,
                List[Tuple[VertexIntervalSerializer.SerializedType, int, int]],
            ]
        ],
    ]

    def serialize(
        ea: SubgoalEdgeActions, he_index: Dict[HalfEdge, int]
    ) -> SerializedType:
        subgoal_actions = []
        for subgoal in ea.targets:
            subgoal_ser = VertexIntervalSerializer.serialize(subgoal)
            subgoal_targets = [
                (
                    VertexIntervalSerializer.serialize(vi),
                    serialize_nav_dir(nd),
                    he_index[he],
                )
                for (vi, nd, he) in ea.targets[subgoal]
            ]
            subgoal_actions.append((subgoal_ser, subgoal_targets))

        return he_index[ea.edge], subgoal_actions

    def deserialize(
        ser: SerializedType, he_index: List[HalfEdge]
    ) -> SubgoalEdgeActions:
        targets: Dict[
            VertexInterval, List[Tuple[VertexInterval, NavigationDirection, HalfEdge]]
        ] = dict()
        for subgoal_vi, subgoal_targets in ser[1]:
            targets[VertexIntervalSerializer.deserialize(subgoal_vi)] = [
                (
                    VertexIntervalSerializer.deserialize(vi),
                    deserialize_nav_dir(nd),
                    he_index[he],
                )
                for (vi, nd, he) in subgoal_targets
            ]

        return SubgoalEdgeActions(he_index[ser[0]], targets)


class SubgoalPolicy:
    def __init__(
        self,
        gridw: PolygonGridWorld,
        ea: Optional[Dict[HalfEdge, SubgoalEdgeActions]] = None,
        subgoals: List[VertexInterval] = None,
        start_edge: Optional[HalfEdge] = None,
        start_point: Optional[Vertex] = None,
        btree: Optional[BackwardReachabilityTree] = None,
        start_leaf: Optional[BackwardReachabilityTreeNode] = None,
    ) -> None:
        self.gridw = gridw
        self.built = True if ea else False
        self.edge_actions: Dict[HalfEdge, SubgoalEdgeActions] = ea if ea else dict()
        self.subgoals: List[VertexInterval] = subgoals or []
        self.current_subgoal = 0
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

        visited_edges = set()
        curr = self.start_leaf
        visited_edges.add(VertexInterval(edge=curr.linked_edge))
        self.subgoals.append(VertexInterval(edge=curr.linked_edge))

        while curr.forward_node and not curr.contains_target:
            if not curr.linked_edge in self.edge_actions:
                self.edge_actions[curr.linked_edge] = SubgoalEdgeActions(
                    curr.linked_edge
                )

            self.edge_actions[curr.linked_edge].add_target(
                curr.forward_node.edge, curr.forward_node.linked_edge, self.subgoals[-1]
            )
            curr = curr.forward_node

            vi_linked_edge = VertexInterval(edge=curr.linked_edge)
            if vi_linked_edge not in visited_edges:
                visited_edges.add(vi_linked_edge)
                self.subgoals.append(vi_linked_edge)

        self.built = True

    def navigate(self, coord: Vertex, edge: HalfEdge) -> Tuple[HalfEdge, HalfEdge]:
        assert self.built
        assert edge in self.edge_actions

        if self.current_subgoal + 1 < len(self.subgoals) and self.subgoals[
            self.current_subgoal + 1
        ] == VertexInterval(edge=edge):
            self.current_subgoal += 1
        return self.edge_actions[edge].navigate(
            coord, self.subgoals[self.current_subgoal]
        )

    def restart(self) -> None:
        assert self.built
        self.current_subgoal = 0
        for _, ea in self.edge_actions.items():
            ea.restart()

    def to_pseudocode(self, filename: str) -> None:
        with open(filename, "w") as f:
            for subgoal in self.subgoals:
                f.writelines([f"Until([{subgoal.start}, {subgoal.end}]):\n"])
                for he in self.edge_actions:
                    if len(self.edge_actions[he].targets.get(subgoal, [])) > 0:
                        f.writelines([f"\t[{he.start}, {he.end}] ->\n"])
                        for vi, dir, e in self.edge_actions[he].targets[subgoal]:
                            f.writelines(
                                [
                                    f"\t\t[{vi.start}, {vi.end}], Preference: {str_nav_dir(dir)}, Edge: [{e.start}, {e.end}]\n"
                                ]
                            )
                f.writelines(["\n"])

            f.writelines(["\tWin!\n"])


class SubgoalPolicySerializer:
    SerializedType = Tuple[
        PolygonGridWorldSerializer.SerializedType,
        List[Tuple[int, SubgoalEdgeActionsSerializer.SerializedType]],
        List[VertexIntervalSerializer.SerializedType],
        int,
        VertexSerializer.SerializedType,
    ]

    def serialize(p: SubgoalPolicy, start_point=Vertex(0, 0)) -> SerializedType:
        gridw_serializer = PolygonGridWorldSerializer(p.gridw)
        gridw_ser = gridw_serializer.serialize()
        eas = [
            (
                gridw_serializer.rev_he_index[he],
                SubgoalEdgeActionsSerializer.serialize(
                    ea, gridw_serializer.rev_he_index
                ),
            )
            for (he, ea) in p.edge_actions.items()
        ]
        subgoals = [
            VertexIntervalSerializer.serialize(subgoal) for subgoal in p.subgoals
        ]
        start_edge = gridw_serializer.rev_he_index[p.start_edge] if p.start_edge else -1
        start_point = VertexSerializer.serialize(start_point)

        return gridw_ser, eas, subgoals, start_edge, start_point

    def deserialize(ser: SerializedType) -> SubgoalPolicy:
        gridw_serializer = PolygonGridWorldSerializer()
        gridw = gridw_serializer.deserialize(ser[0])

        eas = {
            gridw_serializer.half_edges[he]: SubgoalEdgeActionsSerializer.deserialize(
                ea, gridw_serializer.half_edges
            )
            for (he, ea) in ser[1]
        }
        subgoals = [VertexIntervalSerializer.deserialize(subgoal) for subgoal in ser[2]]

        start_edge = gridw_serializer.half_edges[ser[3]] if ser[3] != -1 else None

        return SubgoalPolicy(
            gridw,
            eas,
            subgoals,
            start_edge,
            start_point=VertexSerializer.deserialize(ser[4]),
        )
