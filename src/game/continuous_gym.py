from typing import Any, List, Optional, SupportsFloat, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import drawsvg as dw
from itertools import chain
import random

from src.linpreds import Direction, DirectionSets
from src.polygons import HalfEdge, PolygonGridWorld, Vertex, VertexFloat


class ContinuousReachabilityGridGame(gym.Env):
    def __init__(
        self,
        gridw: PolygonGridWorld,
        start_edge: HalfEdge,
        start_coord: Vertex = VertexFloat(0, 0),
    ) -> None:
        super().__init__()

        self.gridw = gridw
        self.grid_size = gridw.grid_size

        self.start_edge = start_edge
        self.start_coord = start_coord

        self.current_edge = start_edge
        self.current_coord = start_coord

        self.traversed_edges: List[HalfEdge] = []

        self.target_edges = set(
            chain(
                gridw.target.edges_in_polygon(),
                (t.opp for t in gridw.target.edges_in_polygon() if t.opp is not None),
            )
        )
        self.target_coord = gridw.target.middle_of_polygon()

        # observation space is just the coordinates of the agent
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(2,), dtype=np.float32
        )

        # action space is the set of directions agent can move around
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

    def _get_obs(self):
        return np.array(
            [np.float32(self.current_coord.x), np.float32(self.current_coord.y)]
        )

    def _get_info(self):
        return dict()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        self.current_edge = self.start_edge
        self.current_coord = self.start_coord
        self.traversed_edges = []

        return self._get_obs(), self._get_info()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        dir = self._dir_from_action(action[0])
        ncoord, nedge = self._make_transition(dir)

        self.traversed_edges.append(HalfEdge(self.current_coord, ncoord))
        reward = (self.target_coord - self.current_coord).norm() - (
            self.target_coord - ncoord
        ).norm()
        self.current_coord, self.current_edge = ncoord, nedge

        terminated = self.current_edge in self.target_edges
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _validate_direction(self, edge: HalfEdge, dir: VertexFloat) -> VertexFloat:
        x, y = dir.x, dir.y
        acts = DirectionSets[edge.actions] if edge.actions else ()
        if y > 0:
            if not Direction.U in acts:
                y = 0
                x = (
                    2 * self.grid_size
                    if x > 0
                    else (-2 * self.grid_size if x < 0 else 0)
                )
        elif y < 0:
            if not Direction.D in acts:
                y = 0
                x = (
                    2 * self.grid_size
                    if x > 0
                    else (-2 * self.grid_size if x < 0 else 0)
                )

        if x > 0:
            if not Direction.R in acts:
                x = 0
                y = (
                    2 * self.grid_size
                    if y > 0
                    else (-2 * self.grid_size if y < 0 else 0)
                )
        elif x < 0:
            if not Direction.L in acts:
                x = 0
                y = (
                    2 * self.grid_size
                    if y > 0
                    else (-2 * self.grid_size if y < 0 else 0)
                )

        return VertexFloat(x, y)

    def _dir_from_action(self, a: float) -> VertexFloat:
        return VertexFloat(np.cos(2 * np.pi * a), np.sin(2 * np.pi * a)).mult_const(
            2 * self.grid_size
        )

    def _compute_transition(
        self, edge: HalfEdge, dir: VertexFloat
    ) -> Tuple[VertexFloat, HalfEdge]:
        if not edge:
            return None, None
        dir = self._validate_direction(edge, dir)
        if dir.x == 0 and dir.y == 0:
            return self.current_coord, self.current_edge
        dir_edge = HalfEdge(self.current_coord, self.current_coord + dir)
        intersection_vertex = dir_edge.intersects_edge(
            edge, consider_colinear=True, current_coord=self.current_coord
        )
        intersection_edge = edge

        while not intersection_vertex and intersection_edge.next != edge:
            intersection_edge = intersection_edge.next
            intersection_vertex = dir_edge.intersects_edge(
                intersection_edge,
                consider_colinear=True,
                current_coord=self.current_coord,
            )

        if not intersection_vertex:
            return self.current_coord, self.current_edge
        return intersection_vertex, intersection_edge

    def _make_transition(self, dir: VertexFloat) -> Tuple[VertexFloat, HalfEdge]:
        possible_transitions = [self._compute_transition(self.current_edge, dir)]
        if self.current_edge.opp:
            possible_transitions.append(
                self._compute_transition(self.current_edge.opp, dir)
            )

        edge = self.current_edge
        if self.current_coord == self.current_edge.start:
            edge = edge.prev

        if self.current_coord == edge.end:
            possible_transitions.append(self._compute_transition(edge, dir))
            if edge.opp:
                possible_transitions.append(self._compute_transition(edge.opp, dir))

            nedge = edge.next
            while nedge.opp != edge:
                possible_transitions.append(self._compute_transition(nedge, dir))
                if nedge.opp:
                    possible_transitions.append(
                        self._compute_transition(nedge.opp, dir)
                    )
                    nedge = nedge.opp.next
                else:
                    break

        valid_transitions = list(
            filter(
                lambda x: x[0] is not None and x[0] != self.current_coord,
                possible_transitions,
            )
        )
        if not valid_transitions:
            return self.current_coord, self.current_edge
        return random.choice(valid_transitions)

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
