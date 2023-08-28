from manim import *
from src.game.continuous import ContinuousReachabilityGridGame
from src.polygons import PolygonGridWorld, Vertex

from src.grimanim.gridworld import Grid


class PolicyPath(Grid):
    def _construct_policy(
        self,
        gridw: PolygonGridWorld,
        game: ContinuousReachabilityGridGame,
        draw_grid=False,
        animate_grid=False,
        animate_grid_world=False,
        animate_path=True,
    ):
        self._construct_grid_world(
            gridw,
            draw_grid=draw_grid,
            animate_grid=animate_grid,
            animate_grid_world=animate_grid_world,
            entry_point=game.start_coord,
        )

        if not game.traversed_edges:
            return

        path = self._edge_to_line(game.traversed_edges[0])
        for e in game.traversed_edges[1:]:
            path.append_points(self._edge_to_line(e).points)

        path.set_stroke(color=BLUE)
        self._gridworld.add(path)

        if animate_path:
            self.play(
                Create(path),
                run_time=max(len(game.traversed_edges) / 10, 1),
                rate_func=rate_functions.ease_in_sine,
            )
        else:
            self.add(path)
