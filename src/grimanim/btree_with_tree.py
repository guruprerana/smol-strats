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

        # trim tree so that spacing of segments becomes good
        btree.trim(max_depth)

        self.play(self._gridworld.animate.move_to(np.array([-3.5, 0, 0])))

        self._node_to_win_polygon: Dict[
            BackwardReachabilityTreeNode, Optional[Polygon]
        ] = dict()
        self._win_polygons = VGroup()
        current_roots = [root for root in btree.roots]

        prev_segments, prev_segment_texts, prev_y_spaces = self._construct_segments(
            current_roots, 3, -3
        )
        prev_segment_texts.set_x(self._gridworld.get_right()[0] + 1.5)
        self.play(Create(prev_segments))
        self.play(
            *[
                Transform(segment, text)
                for (segment, text) in zip(
                    prev_segments.submobjects, prev_segment_texts.submobjects
                )
            ]
        )
        self.play(prev_segment_texts.animate.set_color(WHITE).set_fill(color=WHITE))

        for _ in range(max_depth):
            new_roots = []
            segments = VGroup()
            segment_texts = VGroup()
            y_spaces: List[Tuple[float, float]] = list()
            for i, root in enumerate(current_roots):
                if not root.backward_nodes:
                    continue
                vertices = [root.edge.start, root.edge.end]
                for node in root.backward_nodes:
                    new_roots.append(node)
                    vertices.extend((node.edge.start, node.edge.end))

                vertices = [np.array([float(v.x), float(v.y)]) for v in vertices]
                hull_vertices = ConvexHull(vertices, qhull_options="QJ").vertices
                vertices = [
                    self._scale_coordinates_2d_floats(
                        self._grid, vertices[i], self._grid_size
                    )
                    for i in hull_vertices
                ]
                win_polygon = Polygon(*vertices).set_fill(color=BLUE, opacity=0.5)
                self._node_to_win_polygon[root] = win_polygon
                self._win_polygons.add(win_polygon)
                self._gridworld.add(win_polygon)

                self.play(
                    GrowFromPoint(
                        win_polygon,
                        np.mean(np.array(self._edge_to_vertices(root.edge)), axis=0),
                    )
                )

                (
                    root_segments,
                    root_segment_texts,
                    root_y_spaces,
                ) = self._construct_segments(
                    root.backward_nodes, prev_y_spaces[i][0], prev_y_spaces[i][1]
                )
                segments.add(*root_segments.submobjects)
                segment_texts.add(*root_segment_texts.submobjects)
                y_spaces.extend(root_y_spaces)

                root_segment_texts.set_x(prev_segment_texts.get_right()[0] + 1.5)

                root_arrows = VGroup()
                for j, node in enumerate(root.backward_nodes):
                    root_text = prev_segment_texts.submobjects[i]
                    node_text = root_segment_texts.submobjects[j]
                    root_arrows.add(
                        Line(
                            root_text.get_right(),
                            node_text.get_left(),
                            buff=SMALL_BUFF,
                        ).set_stroke(color=GRAY, width=0.5)
                    )
                self.play(Create(root_segments))
                self.play(
                    *[
                        Transform(segment, text)
                        for (segment, text) in zip(
                            root_segments.submobjects, root_segment_texts.submobjects
                        )
                    ]
                )
                self.play(
                    Create(root_arrows),
                    root_segment_texts.animate.set_color(WHITE).set_fill(color=WHITE),
                )

            if not new_roots:
                break
            current_roots = new_roots
            prev_segments, prev_segment_texts, prev_y_spaces = (
                segments,
                segment_texts,
                y_spaces,
            )

    def _construct_segments(
        self, nodes: List[BackwardReachabilityTreeNode], y_max, y_min
    ) -> Tuple[VGroup, VGroup, List[Tuple[float, float]]]:
        segments = VGroup()
        texts = VGroup()

        if not nodes:
            return segments, texts

        total_leaves = 0

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

            total_leaves += node.count_leaves()

        texts.arrange(DOWN)

        y_spaces = []
        buff = (y_max - y_min) / total_leaves
        y_space_begin = y_max
        y_space_end = y_space_begin - (buff * nodes[0].count_leaves())
        y_spaces.append((y_space_begin, y_space_end))
        texts.submobjects[0].set_y((y_space_begin + y_space_end) / 2)

        for node, text in zip(nodes[1:], texts.submobjects[1:]):
            y_space_begin, y_space_end = y_space_end, y_space_end - (
                buff * node.count_leaves()
            )
            y_spaces.append((y_space_begin, y_space_end))
            text.set_y((y_space_begin + y_space_end) / 2)

        return segments, texts, y_spaces
