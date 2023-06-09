from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, List, Optional
from linpreds import Direction, DirectionSets
from polygons import HalfEdge, PolygonGridWorld, Vertex, det, dot
from gmpy2 import mpq


def leq(v1: Vertex, v2: Vertex) -> bool:
    """Lexicographic ordering on 2D vertices

    Args:
        v1 (Vertex): first vertex
        v2 (Vertex): second vertex

    Returns:
        bool: true if v1 <= v2 for the lexicographic ordering
    """
    return v1.x < v2.x or (v1.x == v2.x and v1.y <= v2.y)


class VertexInterval:
    def __init__(
        self, start: Vertex, end: Vertex, edge: Optional[HalfEdge] = None
    ) -> None:
        if edge:
            start, end = edge.start, edge.end
        if not leq(start, end):
            start, end = end, start
        self.start: Vertex = start
        self.end: Vertex = end

    def det(self, vi: VertexInterval) -> mpq:
        """Determinant of two vertex intervals interpreted as vectors in 2D space

        Args:
            vi (VertexInterval): other vertex interval

        Returns:
            mpq: determinant of self and vi
        """
        return det(self.end - self.start, vi.end - vi.start)

    def dot(self, vi: VertexInterval) -> mpq:
        """Dot product of two vertex intervals interpreted as vectors in 2D Space

        Args:
            vi (VertexInterval): other vertex interval

        Returns:
            mpq: dot product of self and vi
        """
        return dot(self.end - self.start, vi.end - vi.start)

    def copy(self) -> VertexInterval:
        """Creates a copy of self and returns a new VertexInterval

        Returns:
            VertexInterval: copy of self
        """
        return VertexInterval(self.start, self.end)

    def difference(self, vi: VertexInterval) -> List[VertexInterval]:
        """Performs set difference and removes parts of self which also occur in vi

        Args:
            vi (VertexInterval): vertex interval to remove

        Returns:
            List[VertexInterval]: resulting vertex intervals
            whose union forms the set difference
        """
        # the two vertex intervals should lie along the same line
        assert (
            self.det(vi) == 0
            and det(vi.start - self.start, self.end - self.start) == mpq(0)
            and self.dot(vi) >= mpq(0)
        )
        if leq(vi.end, self.start) or leq(self.end, vi.start):
            return [self.copy()]
        elif leq(vi.start, self.start) and leq(vi.end, self.end):
            return [VertexInterval(vi.end, self.end)]
        elif leq(self.start, vi.start) and leq(self.end, vi.end):
            return [VertexInterval(self.start, vi.start)]
        elif leq(self.start, vi.start) and leq(vi.end, self.end):
            return [
                VertexInterval(self.start, vi.start),
                VertexInterval(vi.end, self.end),
            ]
        else:
            raise ValueError

    def difference_intervals(
        self, vis: Iterable[VertexInterval]
    ) -> List[VertexInterval]:
        res = [self.copy()]
        for vi in vis:
            new_res: List[VertexInterval] = []
            for vi1 in res:
                new_res.extend(vi1.difference(vi))

            res = new_res
        return res

    def intersect_x_bounds(
        self, lower: Optional[mpq] = None, upper: Optional[mpq] = None
    ) -> List[VertexInterval]:
        """Constructs a VertexInterval that contains points from self within
        the lower and upper bounds of the x coordinate

        Args:
            lower (Optional[mpq], optional): lower bound on x coordinate. Defaults to None.
            upper (Optional[mpq], optional): upper bound on x coordinate. Defaults to None.

        Returns:
            List[VertexInterval]: intersection of self within the specified x-coordinate bounds
        """
        if self.start.x == self.end.x:
            if lower <= self.start.x <= upper:
                return [self.copy()]
            else:
                return []

        t_lower = 0
        if lower is not None:
            t_lower = (lower - self.start.x) / (self.end.x - self.start.x)

        t_upper = 1
        if upper is not None:
            t_upper = (upper - self.start.x) / (self.end.x - self.start.x)

        t_lower = max(mpq(0), t_lower)
        t_upper = min(mpq(1), t_upper)

        if t_lower > 1 or t_upper < 0:
            return []

        v_lower = self.start + (self.end - self.start).mult_const(t_lower)
        v_upper = self.start + (self.end - self.start).mult_const(t_upper)

        return [VertexInterval(v_lower, v_upper)]

    def intersect_y_bounds(
        self, lower: Optional[mpq] = None, upper: Optional[mpq] = None
    ) -> List[VertexInterval]:
        """Constructs a VertexInterval that contains points from self within
        the lower and upper bounds of the y coordinate

        Args:
            lower (Optional[mpq], optional): lower bound on y coordinate. Defaults to None.
            upper (Optional[mpq], optional): upper bound on y coordinate. Defaults to None.

        Returns:
            List[VertexInterval]: intersection of self within the specified y-coordinate bounds
        """
        if self.start.y == self.end.y:
            if lower <= self.start.y <= upper:
                return [self.copy()]
            else:
                return []

        t_lower = 0
        if lower is not None:
            t_lower = (lower - self.start.y) / (self.end.y - self.start.y)

        t_upper = 1
        if upper is not None:
            t_upper = (upper - self.start.y) / (self.end.y - self.start.y)

        t_lower = max(mpq(0), t_lower)
        t_upper = min(mpq(1), t_upper)

        if t_lower > 1 or t_upper < 0:
            return []

        v_lower = self.start + (self.end - self.start).mult_const(t_lower)
        v_upper = self.start + (self.end - self.start).mult_const(t_upper)

        return [VertexInterval(v_lower, v_upper)]

    def contains_vertex(self, v: Vertex) -> bool:
        """Checks whether vertex v belongs to the interval

        Args:
            v (Vertex): the vertex to check

        Returns:
            bool: true if v belongs to the vertex interval
        """
        return det(v - self.start, self.end - self.start) == 0 and 0 <= dot(
            v - self.start, self.end - self.start
        ) <= dot(self.end - self.start, self.end - self.start)


class BackwardReachabilityTreeNode:
    def __init__(
        self,
        edge: VertexInterval,
        linked_edge: HalfEdge,
        forward_node: Optional[BackwardReachabilityTreeNode] = None,
        backward_nodes: Optional[List[BackwardReachabilityTreeNode]] = None,
        contains_target: bool = False,
        contains_begin: bool = False,
    ) -> None:
        self.edge = edge
        self.linked_edge = linked_edge
        self.backward_nodes = backward_nodes if backward_nodes else list()
        self.forward_node = forward_node
        self.contains_target = contains_target
        self.contains_begin = (
            contains_begin if contains_begin else edge.contains_vertex(Vertex(0, 0))
        )

    def add_backward_node(self, node: BackwardReachabilityTreeNode) -> None:
        self.backward_nodes.append(node)


class BackwardReachabilityTree:
    def __init__(self, polygrid: PolygonGridWorld) -> None:
        self.polygrid = polygrid

        assert polygrid.target is not None

        self.edges_to_nodes: Dict[
            HalfEdge, List[BackwardReachabilityTreeNode]
        ] = defaultdict(default_factory=lambda _: [])

        def tree_node_from_target_edge(
            target_edge: HalfEdge,
        ) -> BackwardReachabilityTreeNode:
            node = BackwardReachabilityTreeNode(
                VertexInterval(target_edge.start, target_edge.end),
                target_edge,
                contains_target=True,
            )
            self.edges_to_nodes[target_edge] = [node]
            return node

        self.roots = [
            tree_node_from_target_edge(edge)
            for edge in polygrid.target.edges_in_polygon
        ]
        self.leaves = self.roots

    def grow(self) -> bool:
        """Grows the backward reachability tree by extending the leaves
        if the leaves are extensible and have not reached the beginning (0,0) yet.

        Returns:
            bool: true if at least one leaf was extended and false otherwise
        """
        new_leaves = []
        added_nodes = False

        for leaf in self.leaves:
            if leaf.contains_begin or not leaf.linked_edge.opp:
                # we don't want to extend this more
                new_leaves.append(leaf)
                continue

            opp_edge = leaf.linked_edge.opp
            x_bounds = [
                min(opp_edge.start.x, opp_edge.end.x),
                max(opp_edge.start.x, opp_edge.start.x),
            ]
            y_bounds = [
                min(opp_edge.start.y, opp_edge.end.y),
                max(opp_edge.start.y, opp_edge.start.y),
            ]

            if not opp_edge.actions:
                raise ValueError
            for action in DirectionSets[opp_edge.actions]:
                if action == Direction.L:
                    x_bounds[1] = None
                elif action == Direction.R:
                    x_bounds[0] = None
                elif action == Direction.U:
                    y_bounds[0] = None
                elif action == Direction.D:
                    y_bounds[1] = None
                else:
                    raise ValueError

            next_edge = opp_edge.next
            while next_edge != opp_edge:
                vi = VertexInterval(edge=next_edge)
                x_filter = vi.intersect_x_bounds(x_bounds[0], x_bounds[1])
                y_filter: List[VertexInterval] = []
                for vi in x_filter:
                    y_filter.extend(vi.intersect_y_bounds(y_bounds[0], y_bounds[1]))

                for vi in y_filter:
                    unexplored_filter = vi.difference_intervals(
                        [node.edge for node in self.edges_to_nodes[next_edge]]
                    )

                    for new_vi in unexplored_filter:
                        new_node = BackwardReachabilityTreeNode(new_vi, next_edge, leaf)
                        added_nodes = True
                        leaf.add_backward_node(new_node)
                        new_leaves.append(new_node)
                        self.edges_to_nodes[next_edge].append(new_node)
                next_edge = next_edge.next

        self.leaves = new_leaves
        return added_nodes

    def construct_tree(self, max_depth=1000) -> None:
        i = 0
        while i < max_depth and self.grow():
            i += 1
