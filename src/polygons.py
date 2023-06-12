from __future__ import annotations
from typing import Callable, Generator, List, Optional, Set, Union
from gmpy2 import mpq
import drawsvg as dw

from src.linpreds import Direction, DirectionSets, LinearPredicatesGridWorld


class Vertex:
    def __init__(self, x: Union[mpq, int], y: Union[mpq, int]) -> None:
        self.x = mpq(x)
        self.y = mpq(y)

    def __add__(self, v: Vertex) -> Vertex:
        return Vertex(self.x + v.x, self.y + v.y)

    def __neg__(self) -> Vertex:
        return Vertex(-self.x, -self.y)

    def __sub__(self, v: Vertex) -> Vertex:
        return Vertex(self.x - v.x, self.y - v.y)

    def mult_const(self, t: mpq) -> Vertex:
        return Vertex(self.x * t, self.y * t)

    def __eq__(self, v: object) -> bool:
        if not isinstance(v, self.__class__):
            return False
        return self.x == v.x and self.y == v.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Vertex({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


def det(v1: Vertex, v2: Vertex) -> mpq:
    """Determinant of two vertices interpreted as vectors

    Args:
        v1 (Vertex): first vector
        v2 (Vertex): second vector

    Returns:
        mpq: determinant of v1 and v2
    """
    return (v1.x * v2.y) - (v1.y * v2.x)


def dot(v1: Vertex, v2: Vertex) -> mpq:
    """Dot product of two vertices interpreted as vectors

    Args:
        v1 (Vertex): first vector
        v2 (Vertex): second vector

    Returns:
        mpq: dot product of v1 and v2
    """
    return (v1.x * v2.x) + (v1.y * v2.y)


class HalfEdge:
    def __init__(
        self,
        start: Vertex,
        end: Vertex,
        next: Optional[HalfEdge] = None,
        prev: Optional[HalfEdge] = None,
        opp: Optional[HalfEdge] = None,
        actions: Optional[int] = None,
    ) -> None:
        assert start != end
        self.start = start
        self.end = end
        self.next = next
        self.prev = prev
        self.opp = opp
        self.actions = actions

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __repr__(self) -> str:
        return f"HalfEdge({str(self.start)}, {str(self.end)})"

    def edges_in_polygon(self) -> Generator[HalfEdge, None, None]:
        yield self
        edge = self.next
        while edge != self:
            yield edge
            edge = edge.next

    def navigate(self, path: str) -> HalfEdge:
        pt = self
        for p in path:
            if p == "n":
                pt = pt.next
            elif p == "p":
                pt = pt.prev
            elif p == "o":
                if not pt.opp:
                    raise ValueError
                pt = pt.opp
            else:
                raise ValueError
        return pt

    def intersects_edge(self, e: HalfEdge) -> Optional[Vertex]:
        """Determines whether edge intersects another edge e

        Args:
            e (HalfEdge): the other edge to check intersection

        Returns:
            Optional[Vertex]: returns the vertex of intersection if it exists else None
        """
        if self.start in (e.start, e.end):
            return self.start
        elif self.end in (e.start, e.end):
            return self.end

        # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        p, r, q, s = self.start, self.end - self.start, e.start, e.end - e.start

        detrs = det(r, s)

        if detrs == 0:
            # colinear or parallel
            # technically it can intersect a portion of the line but we do not consider this case
            return None

        t = det(q - p, s) / detrs
        u = det(q - p, r) / detrs

        if 0 <= t <= 1 and 0 <= u <= 1:
            return p + r.mult_const(t)
        else:
            # not parallel but do not intersect
            return None

    def add_vertex(self, v: Vertex) -> HalfEdge:
        """Divides the half edge and its opposite into two parts at the vertex v

        Args:
            v (Vertex): vertex at which to divide the edge

        Returns:
            HalfEdge: returns the first half edge after the division
        """
        if v in (self.start, self.end):
            raise ValueError

        p, r = self.start, self.end - self.start
        t = dot(v - p, r) / dot(r, r)
        if det(v - p, r) != 0 or (not (0 <= t <= 1)):
            # v does not lie on the half edge
            raise ValueError

        e1 = HalfEdge(self.start, v, prev=self.prev, actions=self.actions)
        e2 = HalfEdge(v, self.end, next=self.next, prev=e1, actions=self.actions)
        self.prev.next = e1
        self.next.prev = e2
        e1.next = e2

        eopp1, eopp2 = None, None
        if self.opp:
            eopp1 = HalfEdge(
                self.end,
                v,
                prev=self.opp.prev,
                opp=e2,
                actions=self.opp.actions,
            )
            eopp2 = HalfEdge(
                v,
                self.start,
                next=self.opp.next,
                prev=eopp1,
                opp=e1,
                actions=self.opp.actions,
            )

            e1.opp = eopp2
            e2.opp = eopp1
            self.opp.prev.next = eopp1
            self.opp.next.prev = eopp2
            eopp1.next = eopp2

        return e1

    def add_intersection_edge(self, e: HalfEdge) -> None:
        """Given an edge e, recursively adds edges through the whole graph
        along e

        Args:
            e (HalfEdge): edge along which to add edges
        """
        intersection = self.intersects_edge(e)
        if not intersection:
            return

        if intersection == self.start:
            return self.prev.add_intersection_edge(e)

        if intersection != self.end:
            # intersection lies between start and end so we add a vertex there
            e1 = self.add_vertex(intersection)
            return e1.add_intersection_edge(e)

        # determine next vertex of intersection in the polygon
        e_int = self.next.next
        intersection1 = None
        while e_int != self:
            intersection1 = e_int.intersects_edge(e)
            if intersection1:
                break
            e_int = e_int.next

        if e_int == self or intersection1 == e_int.start:
            raise ValueError

        # intersection1 is either in between e1 or at its end
        if intersection1 != e_int.end:
            e_int = e_int.add_vertex(intersection1)

        # now we need to add halfedges between self.end and e1.end
        divedge = HalfEdge(self.end, e_int.end, e_int.next, self, actions=self.actions)
        divedge_opp = HalfEdge(
            e_int.end, self.end, self.next, e_int, divedge, actions=self.actions
        )
        divedge.opp = divedge_opp

        self.next.prev = divedge_opp
        e_int.next.prev = divedge

        self.next = divedge
        e_int.next = divedge_opp

        if e_int.opp:
            return e_int.opp.prev.add_intersection_edge(e)

    def add_diagonal(self, v: Vertex) -> None:
        """Adds a diagonal edge between self.end and v which is assumed to be an endpoint of
        self's successors

        Args:
            v (Vertex): the end point of the diagonal
        """
        if v in (self.start, self.end):
            raise ValueError

        e_int = self.next
        while e_int != self:
            if e_int.end == v:
                break
            e_int = e_int.next

        if e_int in (self.prev, self, self.next):
            raise ValueError

        divedge = HalfEdge(self.end, e_int.end, e_int.next, self, actions=self.actions)
        divedge_opp = HalfEdge(
            e_int.end, self.end, self.next, e_int, divedge, actions=self.actions
        )
        divedge.opp = divedge_opp

        self.next.prev = divedge_opp
        e_int.next.prev = divedge

        self.next = divedge
        e_int.next = divedge_opp

    def determine_side(self, e: HalfEdge) -> bool:
        """Similar to LinearPredicate.determine_side, determines the side
        of a whole edge (self) with respect to e

        Args:
            e (HalfEdge): linear predicate along which to determine side

        Returns:
            bool: true iff all the points along the edge lie on the 'positive' side of e
        """
        d1 = det(self.start - e.start, e.end - e.start)
        d2 = det(self.end - e.start, e.end - e.start)

        # determinant is convex with respect to both arguments
        # so if both are pos or neg, then all points in between are pos or neg resp.
        if (d1 > 0 and d2 >= 0) or (d1 >= 0 and d2 > 0):
            return True
        elif (d1 < 0 and d2 <= 0) or (d1 <= 0 and d2 < 0):
            return False
        elif d1 == d2 == 0:
            if det(self.end - self.start, self.next.end - self.next.start) == 0:
                # next edge is colinear to this one, we don't want to deal with this situation right now
                raise ValueError
            return self.next.determine_side(e)
        else:
            raise ValueError

    def assign_action_polygon(self, actions: int) -> None:
        assert 0 <= actions < 1 << 4
        for edge in self.edges_in_polygon():
            edge.actions = actions


class PolygonGridWorld:
    def __init__(
        self,
        root: Optional[HalfEdge] = None,
        target: Optional[HalfEdge] = None,
        grid_size: Optional[int] = None,
    ) -> None:
        self.root = root
        self.target = target
        self.grid_size = grid_size

    def traverse(self, f: Callable[[HalfEdge], None]) -> None:
        """Calls the function f on each HalfEdge in the grid world.
        Performs a simple depth-first traversal

        Args:
            f (Callable[[HalfEdge], None]): the function which is called on each HalfEdge
        """
        visited: Set[HalfEdge] = set()
        to_visit = [self.root]

        while len(to_visit) > 0:
            e = to_visit.pop()
            if e in visited:
                continue

            f(e)
            visited.add(e)
            if e.opp and e.opp not in visited:
                to_visit.append(e.opp)

            e1 = e.next
            while e1 != e:
                if e1 not in visited:
                    f(e1)
                    visited.add(e1)

                if e1.opp and e1.opp not in visited:
                    to_visit.append(e1.opp)
                e1 = e1.next

    def traverse_polygons(self, f: Callable[[HalfEdge], None]) -> None:
        """Calls the function f exactly once on a certain HalfEdge in each polygon
        in the grid world.

        Args:
            f (Callable[[HalfEdge], None]): the function which is called on the HalfEdges
        """
        visited: Set[HalfEdge] = set()
        to_visit = [self.root]

        while len(to_visit) > 0:
            e = to_visit.pop()
            if e in visited:
                continue

            f(e)
            visited.add(e)
            if e.opp and e.opp not in visited:
                to_visit.append(e.opp)

            e1 = e.next
            while e1 != e:
                visited.add(e1)

                if e1.opp and e1.opp not in visited:
                    to_visit.append(e1.opp)

                e1 = e1.next

    def draw(
        self,
        filename="polygon-grid.png",
        drawing: Optional[dw.Drawing] = None,
        scale=30,
        p=20,
        dir_line_width=30,
        save=True,
    ) -> None:
        grid_size = self.grid_size if self.grid_size is not None else 1000
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
        # background
        d.append(
            dw.Rectangle(
                0,
                0,
                (grid_size + 1) * scale + p,
                (grid_size + 1) * scale + p,
                fill="black",
            )
        )

        def draw_edge(e: HalfEdge) -> None:
            (x1, y1), (x2, y2) = (e.start.x, e.start.y), (e.end.x, e.end.y)
            d.append(
                dw.Line(
                    float(p + scale * x1),
                    float(p + scale * y1),
                    float(p + scale * x2),
                    float(p + scale * y2),
                    stroke_width=2,
                    stroke="magenta",
                )
            )

        def draw_directions(e: HalfEdge) -> None:
            if not e.actions or dir_line_width == 0:
                return

            total = e.start
            n = 1

            e1 = e.next
            while e1 != e:
                total += e1.start
                n += 1
                e1 = e1.next

            # we want to draw the directions at the center of the polygon
            # dir_line_width = 30
            x, y = (total.x / n, total.y / n)
            x_fig, y_fig = float(p + (x * scale)), float(p + (y * scale))

            for dir in DirectionSets[e.actions]:
                if dir == Direction.L:
                    d.append(
                        dw.Line(
                            x_fig,
                            y_fig,
                            x_fig - dir_line_width,
                            y_fig,
                            stroke="green",
                            stroke_width=2,
                        )
                    )
                elif dir == Direction.R:
                    d.append(
                        dw.Line(
                            x_fig,
                            y_fig,
                            x_fig + dir_line_width,
                            y_fig,
                            stroke="green",
                            stroke_width=2,
                        )
                    )
                elif dir == Direction.U:
                    d.append(
                        dw.Line(
                            x_fig,
                            y_fig,
                            x_fig,
                            y_fig + dir_line_width,
                            stroke="green",
                            stroke_width=2,
                        )
                    )
                elif dir == Direction.D:
                    d.append(
                        dw.Line(
                            x_fig,
                            y_fig,
                            x_fig,
                            y_fig - dir_line_width,
                            stroke="green",
                            stroke_width=2,
                        )
                    )
                else:
                    raise ValueError

                d.append(dw.Circle(x_fig, y_fig, 2, stroke="white", fill="white"))

        self.traverse(draw_edge)
        self.traverse_polygons(draw_directions)
        if save:
            d.save_png(filename)


def simple_rectangular_polygon_gridw(size=1000) -> PolygonGridWorld:
    outer_edge1 = HalfEdge(Vertex(0, 0), Vertex(size, 0))
    outer_edge2 = HalfEdge(Vertex(size, 0), Vertex(size, size), prev=outer_edge1)
    outer_edge1.next = outer_edge2
    outer_edge3 = HalfEdge(Vertex(size, size), Vertex(0, size), prev=outer_edge2)
    outer_edge2.next = outer_edge3
    outer_edge4 = HalfEdge(
        Vertex(0, size), Vertex(0, 0), next=outer_edge1, prev=outer_edge3
    )
    outer_edge3.next = outer_edge4
    outer_edge1.prev = outer_edge4

    grid_world = PolygonGridWorld(outer_edge1, grid_size=size)

    return grid_world


def polygons_from_linpreds(lingrid: LinearPredicatesGridWorld) -> PolygonGridWorld:
    size = lingrid.predicate_grid_size
    outer_edge1 = HalfEdge(Vertex(0, 0), Vertex(size, 0))
    outer_edge2 = HalfEdge(Vertex(size, 0), Vertex(size, size), prev=outer_edge1)
    outer_edge1.next = outer_edge2
    outer_edge3 = HalfEdge(Vertex(size, size), Vertex(0, size), prev=outer_edge2)
    outer_edge2.next = outer_edge3
    outer_edge4 = HalfEdge(
        Vertex(0, size), Vertex(0, 0), next=outer_edge1, prev=outer_edge3
    )
    outer_edge3.next = outer_edge4
    outer_edge1.prev = outer_edge4

    outer_edges = [outer_edge1, outer_edge2, outer_edge3, outer_edge4]
    preds_half_edges: List[HalfEdge] = []

    for pred in lingrid.preds:
        pred_edge = HalfEdge(Vertex(*pred.p1), Vertex(*pred.p2))
        preds_half_edges.append(pred_edge)

        intersecting_edges: List[HalfEdge] = []
        new_outer_edges: List[HalfEdge] = []

        for edge in outer_edges:
            intersection = edge.intersects_edge(pred_edge)
            if not intersection or intersection in (edge.start, edge.end):
                if intersection == edge.end:
                    intersecting_edges.append(edge)
                new_outer_edges.append(edge)
                continue

            # add vertex
            e1 = edge.add_vertex(intersection)
            intersecting_edges.append(e1)
            new_outer_edges.extend((e1, e1.next))

        outer_edges = new_outer_edges

        if len(intersecting_edges) > 0:
            intersecting_edges[0].add_intersection_edge(pred_edge)

    root = outer_edges[0]
    grid_world = PolygonGridWorld(root, grid_size=size)

    target_region = lingrid._region_of_point((size, size), None)
    target_edges: List[HalfEdge] = []

    def assign_actions(e: HalfEdge) -> None:
        actions_index = sum(
            1 << i for (i, p) in enumerate(preds_half_edges) if e.determine_side(p)
        )
        if actions_index == target_region:
            target_edges.append(e)
        e.actions = lingrid.actions[actions_index]

    grid_world.traverse(assign_actions)

    if len(target_edges) == 0:
        raise ValueError

    grid_world.target = target_edges[0]
    return grid_world
