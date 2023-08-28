from typing import Dict, List, Optional, Tuple
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

        self.play(self._gridworld.animate.move_to(np.array([-3, 0, 0])))

        self._node_to_win_polygon: Dict[
            BackwardReachabilityTreeNode, Optional[Polygon]
        ] = dict()
        self._win_polygons = VGroup()
        current_roots = [root for root in btree.roots]

        prev_segments, prev_segment_texts = self._construct_segments(current_roots)
        prev_segment_texts.next_to(self._gridworld, RIGHT, buff=0.5)
        self.play(Create(prev_segments))
        self.play(
            *[
                Transform(segment, text)
                for (segment, text) in zip(
                    prev_segments.submobjects, prev_segment_texts.submobjects
                )
            ]
        )
        self.play(prev_segment_texts.animate.set_color(WHITE))

        for _ in range(max_depth):
            new_roots = []
            current_polygons: List[
                Tuple[Polygon, BackwardReachabilityTreeNode, Mobject]
            ] = []
            for i, root in enumerate(current_roots):
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
                current_polygons.append(
                    (win_polygon, root, prev_segment_texts.submobjects[i])
                )
                self._node_to_win_polygon[root] = win_polygon
                self._win_polygons.add(win_polygon)
                self._gridworld.add(win_polygon)

            segments, segment_texts = self._construct_segments(new_roots)
            segment_texts.next_to(prev_segment_texts, RIGHT, buff=0.5)
            segment_index = 0

            for win_polygon, root, root_text in current_polygons:
                self.play(
                    GrowFromPoint(
                        win_polygon,
                        np.mean(np.array(self._edge_to_vertices(root.edge)), axis=0),
                    )
                )

                root_segments = VGroup()
                root_segments_texts = VGroup()
                root_arrows = VGroup()
                for node in root.backward_nodes:
                    root_segments.add(segments[segment_index])
                    root_segments_texts.add(segment_texts[segment_index])
                    root_arrows.add(
                        Line(
                            root_text.get_right(),
                            segment_texts[segment_index].get_left(),
                            buff=SMALL_BUFF,
                        ).set_stroke(color=GRAY, width=0.5)
                    )

                    segment_index += 1

                self.play(Create(root_segments))
                self.play(
                    *[
                        Transform(segment, text)
                        for (segment, text) in zip(
                            root_segments.submobjects, root_segments_texts.submobjects
                        )
                    ]
                )
                self.play(
                    Create(root_arrows), root_segments_texts.animate.set_color(WHITE)
                )

            if not new_roots:
                break
            current_roots = new_roots
            prev_segments, prev_segment_texts = segments, segment_texts

    def _construct_segments(
        self, nodes: List[BackwardReachabilityTreeNode]
    ) -> Tuple[VGroup, VGroup]:
        segments = VGroup()
        texts = VGroup()

        for node in nodes:
            segment_vertices = self._edge_to_vertices(node.edge)
            segment = (
                Line(*segment_vertices)
                .set_stroke(YELLOW)
                .add(
                    Dot(segment_vertices[0], color=YELLOW).scale(0.5),
                    Dot(segment_vertices[1], color=YELLOW).scale(0.5),
                )
            )
            segments.add(segment)

            texts.add(
                Text(f"[{node.edge.start}, {node.edge.end}]", color=YELLOW).scale(0.3)
            )

        texts.arrange(DOWN)

        buff = 6 / len(nodes)
        start = 3 - (buff / 2)

        for i, text in enumerate(texts.submobjects):
            text.set_y(start - (i * buff))

        return segments, texts
