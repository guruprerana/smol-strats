from __future__ import annotations
from collections import defaultdict
import math
from typing import Dict, Iterable, List, Optional, Tuple
from src.policy import BasePolicy
from src.linpreds import Direction, DirectionSets
from src.polygons import HalfEdge, PolygonGridWorld, Vertex, VertexSerializer, det, dot
from gmpy2 import mpq, to_binary, from_binary
import drawsvg as dw


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
        self, start: Vertex = None, end: Vertex = None, edge: Optional[HalfEdge] = None
    ) -> None:
        if edge:
            start, end = edge.start, edge.end
        if not leq(start, end):
            start, end = end, start
        self.start: Vertex = start
        self.end: Vertex = end

    def __repr__(self) -> str:
        return f"VertexInterval({self.start}, {self.end})"

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return self.start == o.start and self.end == o.end

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
        if leq(vi.start, self.start) and leq(self.end, vi.end):
            return []
        elif leq(vi.end, self.start) or leq(self.end, vi.start):
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
            if (lower is None or lower <= self.start.x) and (
                upper is None or self.start.x <= upper
            ):
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
        start = min(self.start, self.end, key=lambda v: v.y)
        end = max(self.start, self.end, key=lambda v: v.y)
        if start.y == end.y:
            if (lower is None or lower <= start.y) and (
                upper is None or start.y <= upper
            ):
                return [self.copy()]
            else:
                return []

        t_lower = 0
        if lower is not None:
            t_lower = (lower - start.y) / (end.y - start.y)

        t_upper = 1
        if upper is not None:
            t_upper = (upper - start.y) / (end.y - start.y)

        t_lower = max(mpq(0), t_lower)
        t_upper = min(mpq(1), t_upper)

        if t_lower > 1 or t_upper < 0:
            return []

        v_lower = start + (end - start).mult_const(t_lower)
        v_upper = start + (end - start).mult_const(t_upper)

        return [VertexInterval(v_lower, v_upper)]

    def contains_vertex(self, v: Vertex) -> bool:
        """Checks whether vertex v belongs to the interval

        Args:
            v (Vertex): the vertex to check

        Returns:
            bool: true if v belongs to the vertex interval
        """
        if self.end == self.start:
            return self.start == v
        return det(v - self.start, self.end - self.start) == 0 and 0 <= dot(
            v - self.start, self.end - self.start
        ) <= dot(self.end - self.start, self.end - self.start)

    def inside_triangle(self, triangle: Tuple[Vertex, Vertex, Vertex]) -> bool:
        return self.start.inside_triangle(triangle) and self.end.inside_triangle(
            triangle
        )


class VertexIntervalSerializer:
    SerializedType = Tuple[
        VertexSerializer.SerializedType, VertexSerializer.SerializedType
    ]

    def serialize(vi: VertexInterval) -> SerializedType:
        return VertexSerializer.serialize(vi.start), VertexSerializer.serialize(vi.end)

    def deserialize(ser: SerializedType) -> VertexInterval:
        return VertexInterval(
            VertexSerializer.deserialize(ser[0]), VertexSerializer.deserialize(ser[1])
        )


class BackwardReachabilityTreeNode:
    def __init__(
        self,
        edge: VertexInterval,
        linked_edge: HalfEdge,
        depth: int,
        forward_node: Optional[BackwardReachabilityTreeNode] = None,
        backward_nodes: Optional[List[BackwardReachabilityTreeNode]] = None,
        contains_target: bool = False,
        contains_begin: bool = None,
    ) -> None:
        self.edge = edge
        self.linked_edge = linked_edge
        self.depth = depth
        self.backward_nodes = backward_nodes if backward_nodes is not None else list()
        self.forward_node = forward_node
        self.contains_target = contains_target
        self.contains_begin = (
            contains_begin
            if contains_begin is not None
            else edge.contains_vertex(Vertex(0, 0))
        )
        self._count_leaves = None

    def add_backward_node(self, node: BackwardReachabilityTreeNode) -> None:
        self.backward_nodes.append(node)

    def __repr__(self) -> str:
        return f"Node({self.edge.start}, {self.edge.end} -> {str(self.linked_edge)})"

    def print_str(self) -> List[str]:
        res = [repr(self)]
        for node in self.backward_nodes:
            res.extend(f"\t{child_str}" for child_str in node.print_str())

        return res

    def count_leaves(self) -> int:
        if self._count_leaves is not None:
            return self._count_leaves

        if not self.backward_nodes:
            self._count_leaves = 1
        else:
            self._count_leaves = 0
            for node in self.backward_nodes:
                self._count_leaves += node.count_leaves()

        return self._count_leaves

    def trim(self, max_depth: int) -> None:
        if self.depth >= max_depth:
            self.backward_nodes = []
        else:
            for node in self.backward_nodes:
                node.trim(max_depth)


class BackwardReachabilityTree(BasePolicy):
    def __init__(self, polygrid: PolygonGridWorld, begin_point: Vertex = None) -> None:
        self.polygrid = polygrid
        self.begin_point = begin_point

        assert polygrid.target is not None

        self.edges_to_nodes: Dict[
            VertexInterval, List[BackwardReachabilityTreeNode]
        ] = dict()

        def tree_node_from_target_edge(
            target_edge: HalfEdge,
        ) -> BackwardReachabilityTreeNode:
            vi_target_edge = VertexInterval(edge=target_edge)
            node = BackwardReachabilityTreeNode(
                vi_target_edge,
                target_edge,
                0,
                contains_target=True,
            )
            self.edges_to_nodes[vi_target_edge] = [node]
            return node

        self.roots = [
            tree_node_from_target_edge(edge)
            for edge in polygrid.target.edges_in_polygon()
        ]
        self.leaves = self.roots
        self.all_leaves = set(self.leaves)

    def least_depth_begin_leaf(self) -> Optional[BackwardReachabilityTreeNode]:
        begin_leaves = list(filter(lambda x: x.contains_begin, self.all_leaves))
        return min(begin_leaves, key=lambda x: x.depth, default=None)

    def max_depth_leaf(self) -> BackwardReachabilityTreeNode:
        return max(self.all_leaves, key=lambda leaf: leaf.depth)

    def trim(self, max_depth: int) -> None:
        for root in self.roots:
            root.trim(max_depth)

    def grow(self) -> bool:
        """Grows the backward reachability tree by extending the leaves
        if the leaves are extensible and have not reached the beginning (0,0) yet.

        Returns:
            bool: true if at least one leaf was extended and false otherwise
        """
        new_leaves = []
        added_nodes = False

        for leaf in self.leaves:
            extended_leaf = False
            if leaf.contains_begin or not leaf.linked_edge.opp:
                # we don't want to extend this more
                new_leaves.append(leaf)
                continue

            opp_edge = leaf.linked_edge.opp
            x_bounds = [
                min(leaf.edge.start.x, leaf.edge.end.x),
                max(leaf.edge.start.x, leaf.edge.end.x),
            ]
            y_bounds = [
                min(leaf.edge.start.y, leaf.edge.end.y),
                max(leaf.edge.start.y, leaf.edge.end.y),
            ]

            allowed_actions = (
                DirectionSets[opp_edge.actions] if opp_edge.actions else []
            )
            for action in allowed_actions:
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

            for next_edge in opp_edge.edges_in_polygon():
                vi_next_edge = VertexInterval(edge=next_edge)
                x_filter = vi_next_edge.intersect_x_bounds(x_bounds[0], x_bounds[1])
                y_filter: List[VertexInterval] = []
                for vi in x_filter:
                    y_filter.extend(vi.intersect_y_bounds(y_bounds[0], y_bounds[1]))

                if vi_next_edge not in self.edges_to_nodes:
                    self.edges_to_nodes[vi_next_edge] = []

                for vi in y_filter:
                    unexplored_filter = vi.difference_intervals(
                        [node.edge for node in self.edges_to_nodes[vi_next_edge]]
                    )

                    for new_vi in unexplored_filter:
                        # we need final filter to remove from triangle above or below the interval
                        vi_to_add = new_vi
                        if self._inside_unreachable_triangle(
                            new_vi, leaf.edge, x_bounds, y_bounds, allowed_actions
                        ):
                            if leaf.edge.contains_vertex(new_vi.start):
                                vi_to_add = VertexInterval(new_vi.start, new_vi.start)
                            elif leaf.edge.contains_vertex(new_vi.end):
                                vi_to_add = VertexInterval(new_vi.end, new_vi.end)
                            else:
                                continue
                        contains_begin = (
                            vi_to_add.contains_vertex(self.begin_point)
                            if self.begin_point is not None
                            else False
                        )
                        new_node = BackwardReachabilityTreeNode(
                            vi_to_add,
                            next_edge,
                            leaf.depth + 1,
                            leaf,
                            contains_begin=contains_begin,
                        )
                        added_nodes = True
                        leaf.add_backward_node(new_node)
                        new_leaves.append(new_node)
                        extended_leaf = True
                        self.all_leaves.add(new_node)
                        self.edges_to_nodes[vi_next_edge].append(new_node)

            if extended_leaf:
                self.all_leaves.remove(leaf)
        self.leaves = new_leaves
        return added_nodes

    def _inside_unreachable_triangle(
        self,
        vi: VertexInterval,
        edge: VertexInterval,
        x_bounds: Tuple[Optional[mpq], Optional[mpq]],
        y_bounds: Tuple[Optional[mpq], Optional[mpq]],
        allowed_actions: List[Direction],
    ) -> bool:
        if edge.contains_vertex(vi.start) and edge.contains_vertex(vi.end):
            return False
        if not allowed_actions:
            # at most one of the vertices is on the edge
            return True
        if (Direction.L in allowed_actions and Direction.R in allowed_actions) or (
            Direction.U in allowed_actions and Direction.D in allowed_actions
        ):
            return False
        edge_slope = edge.end - edge.start
        if edge_slope.x == 0 or edge_slope.y == 0:
            return False

        edge_slope = edge_slope.y / edge_slope.x
        if edge_slope > 0:
            if (Direction.D in allowed_actions and Direction.L in allowed_actions) or (
                Direction.U in allowed_actions and Direction.R in allowed_actions
            ):
                return False
        if edge_slope < 0:
            if (Direction.D in allowed_actions and Direction.R in allowed_actions) or (
                Direction.U in allowed_actions and Direction.L in allowed_actions
            ):
                return False

        x1, x2 = x_bounds[0], x_bounds[1]
        y1, y2 = y_bounds[0], y_bounds[1]
        points = ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
        triangle = list(
            filter(lambda p: (p[0] is not None) and (p[1] is not None), points)
        )
        triangle = [Vertex(x, y) for (x, y) in triangle]
        triangle.extend((edge.start, edge.end))
        triangle = tuple(set(triangle))
        assert len(triangle) == 3
        return vi.inside_triangle(triangle)

    def construct_tree(self, max_depth=100) -> None:
        i = 0
        while i < max_depth and self.grow():
            i += 1

    def draw(
        self,
        filename="backward-graph.png",
        drawing: Optional[dw.Drawing] = None,
        scale=30,
        p=20,
        save=True,
        start_point: Vertex = None,
    ) -> None:
        grid_size = (
            self.polygrid.grid_size if self.polygrid.grid_size is not None else 1000
        )
        d = (
            drawing
            if drawing is not None
            else dw.Drawing(
                (grid_size + 1) * scale + p,
                (grid_size + 1) * scale + p,
                id_prefix="grid",
                context=dw.Context(invert_y=True),
            )
        )

        self.polygrid.draw(
            None,
            d,
            dir_line_width=0,
            scale=scale,
            p=p,
            save=False,
            start_point=start_point,
        )

        def draw_edge(e: VertexInterval, color="white") -> None:
            (x1, y1), (x2, y2) = (e.start.x, e.start.y), (e.end.x, e.end.y)
            d.append(
                dw.Line(
                    float(p + scale * x1),
                    float(p + scale * y1),
                    float(p + scale * x2),
                    float(p + scale * y2),
                    stroke_width=2,
                    stroke=color,
                )
            )

        def draw_link(e1: VertexInterval, e2: VertexInterval) -> None:
            draw_edge(
                VertexInterval(
                    (e1.start + e1.end).mult_const(mpq(1, 2)),
                    (e2.start + e2.end).mult_const(mpq(1, 2)),
                ),
                color="green",
            )

        to_draw = [node for node in self.roots]
        while len(to_draw) > 0:
            node = to_draw.pop()
            draw_edge(node.edge)
            for node1 in node.backward_nodes:
                to_draw.append(node1)
                draw_link(node.edge, node1.edge)

        if save:
            d.save_png(filename)

    def print(self) -> None:
        for root in self.roots:
            print("\n".join(root.print_str()))
