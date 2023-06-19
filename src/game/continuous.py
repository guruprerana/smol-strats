from typing import List, Optional, Tuple
import drawsvg as dw
from src.linpreds import Direction, DirectionSets
from src.policy import BasePolicy
from src.polygons import HalfEdge, PolygonGridWorld, Vertex


class ContinuousReachabilityGridGame:
    def __init__(
        self,
        gridw: PolygonGridWorld,
        start_edge: HalfEdge,
        start_coord: Vertex = Vertex(0, 0),
    ) -> None:
        self.gridw = gridw
        self.start_edge = start_edge
        self.current_edge = start_edge
        self.start_coord = start_coord
        self.current_coord = start_coord
        self.traversed_edges: List[HalfEdge] = []

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

    def step(
        self, direction: HalfEdge, target_e: HalfEdge
    ) -> Tuple[Vertex, Optional[HalfEdge], bool]:
        assert self.current_coord == direction.start
        self._validate_direction(direction)
        assert target_e in map(lambda x: x.opp, self.current_edge.edges_in_polygon())
        assert target_e.contains_vertex(direction.end)

        self.current_coord = direction.end
        self.current_edge = target_e

        self.traversed_edges.append(HalfEdge(direction.start, direction.end))

        if self.current_edge in self.target_edges:
            return self.current_coord, self.current_edge, True

        return self.current_coord, self.current_edge, False

    def run(self, policy: BasePolicy, max_iter=1000) -> bool:
        target_reached = self.current_edge in self.target_edges
        for _ in range(max_iter):
            dir_e, target_e = policy.navigate(self.current_coord, self.current_edge)
            _, _, target_reached = self.step(dir_e, target_e)
            if target_reached:
                break

        return target_reached

    def draw(
        self,
        filename="policy-path.png",
        drawing: Optional[dw.Drawing] = None,
        scale=30,
        p=20,
        save=True,
    ) -> None:
        grid_size = self.gridw.grid_size if self.gridw.grid_size is not None else 1000
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

        self.gridw.draw(None, d, dir_line_width=0, save=False)

        def draw_edge(e: HalfEdge, color="turquoise") -> None:
            (x1, y1), (x2, y2) = (e.start.x, e.start.y), (e.end.x, e.end.y)
            d.append(
                dw.Line(
                    float(p + scale * x1),
                    float(p + scale * y1),
                    float(p + scale * x2),
                    float(p + scale * y2),
                    stroke_width=4,
                    stroke=color,
                )
            )

        list(map(lambda x: draw_edge(x), self.traversed_edges))

        if save:
            d.save_png(filename)
