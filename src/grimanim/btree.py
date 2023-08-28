from typing import Dict, Optional
from manim import *
from scipy.spatial import ConvexHull
from src.backward_reachability import (
    BackwardReachabilityTree,
    BackwardReachabilityTreeNode,
)
from src.game.continuous import ContinuousReachabilityGridGame
from src.polygons import PolygonGridWorld, Vertex

from src.grimanim.gridworld import Grid


class BTree(Grid):
    def _construct_btree(
        self,
        gridw: PolygonGridWorld,
        game: ContinuousReachabilityGridGame,
        btree: BackwardReachabilityTree,
        draw_grid=False,
        animate_grid=False,
        animate_grid_world=False,
        max_depth=3,
    ):
        self._construct_grid_world(
            gridw,
            draw_grid=draw_grid,
            animate_grid=animate_grid,
            animate_grid_world=animate_grid_world,
            entry_point=game.start_coord,
        )

        self._node_to_win_polygon: Dict[
            BackwardReachabilityTreeNode, Optional[Polygon]
        ] = dict()
        self._win_polygons = VGroup()
        current_roots = [root for root in btree.roots]

        for _ in range(max_depth):
            new_roots = []
            current_polygons = []
            for root in current_roots:
                vertices = [root.edge.start, root.edge.end]
                for node in root.backward_nodes:
                    new_roots.append(node)
                    vertices.extend((node.edge.start, node.edge.end))

                if not len(vertices) > 2:
                    continue

                vertices = [np.array([float(v.x), float(v.y)]) for v in vertices]
                hull_vertices = ConvexHull(vertices, qhull_options="QJ").vertices
                vertices = [
                    self._scale_coordinates_2d_floats(
                        self._grid, vertices[i], self._grid_size
                    )
                    for i in hull_vertices
                ]
                win_polygon = Polygon(*vertices).set_fill(color=BLUE, opacity=0.5)
                current_polygons.append((win_polygon, root))
                self._node_to_win_polygon[root] = win_polygon
                self._win_polygons.add(win_polygon)
                self._gridworld.add(win_polygon)

            if current_polygons:
                self.play(
                    *[
                        GrowFromPoint(
                            win_polygon,
                            np.mean(
                                np.array(self._edge_to_vertices(root.edge)), axis=0
                            ),
                        )
                        for (win_polygon, root) in current_polygons
                    ]
                )

            if not new_roots:
                break
            current_roots = new_roots
