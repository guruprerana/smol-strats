from manim import *
from typing import List, Tuple, Union
from gmpy2 import mpq
import numpy as np
from src.backward_reachability import VertexInterval
from src.linpreds import Direction, DirectionSets
from src.polygons import HalfEdge, PolygonGridWorld, Vertex
from src.grimanim.colors import *


class Grid(Scene):
    def _construct_grid_lines(self, grid: Square, n: int) -> List[Line]:
        lines = []
        vertices = grid.get_vertices()
        bl, br, ur, ul = vertices[2], vertices[3], vertices[0], vertices[1]

        for i in range(0, n):
            r = i / (n - 1)
            lines.append(
                Line(bl + ((br - bl) * r), ul + ((ur - ul) * r)).set_stroke(width=0.5)
            )
            lines.append(
                Line(bl + ((ul - bl) * r), br + ((ur - br) * r)).set_stroke(width=0.5)
            )

        return lines

    def _scale_coordinates(
        self, grid: Square, coord: Vertex, grid_size: int
    ) -> np.ndarray:
        vertices = grid.get_vertices()
        bl, br, ur, ul = vertices[2], vertices[3], vertices[0], vertices[1]

        x, y = coord.x, coord.y

        cx = (br - bl) * float(x / (grid_size - 1))
        cy = (ul - bl) * float(y / (grid_size - 1))

        return bl + cx + cy

    def _scale_coordinates_2d_floats(
        self, grid: Square, coord: Tuple[float, float], grid_size: int
    ) -> np.ndarray:
        vertices = grid.get_vertices()
        bl, br, ur, ul = vertices[2], vertices[3], vertices[0], vertices[1]

        x, y = coord[0], coord[1]

        cx = (br - bl) * float(x / (grid_size - 1))
        cy = (ul - bl) * float(y / (grid_size - 1))

        return bl + cx + cy

    def _construct_grid(
        self, grid: Square, gridw: PolygonGridWorld, animate_grid=True
    ) -> List[Line]:
        grid_size = gridw.grid_size + 1
        grid_lines = self._construct_grid_lines(grid, grid_size)
        if animate_grid:
            self.play(*[Create(line) for line in grid_lines])
            self.play(*[line.animate.set_opacity(0.3) for line in grid_lines])
        else:
            for line in grid_lines:
                line.set_opacity(0.3)
            self.add(*grid_lines)

        self._gridlines.add(*grid_lines)
        self._gridworld.add(*grid_lines)
        return grid_lines

    def _edge_to_line(self, edge: Union[HalfEdge, VertexInterval]) -> Line:
        return Line(
            self._scale_coordinates(self._grid, edge.start, self._grid_size),
            self._scale_coordinates(self._grid, edge.end, self._grid_size),
        )

    def _edge_to_vertices(self, edge: VertexInterval) -> Tuple[np.ndarray, np.ndarray]:
        return self._scale_coordinates(
            self._grid, edge.start, self._grid_size
        ), self._scale_coordinates(self._grid, edge.end, self._grid_size)

    def _construct_grid_world(
        self,
        gridw: PolygonGridWorld,
        draw_grid=True,
        animate_grid=True,
        arrow_length=0.3,
        animate_grid_world=True,
        entry_point=Vertex(0, 0),
    ) -> None:
        # because we count 0 in grid_size
        grid_size = gridw.grid_size + 1
        self._grid_size = grid_size
        grid = Square(side_length=6)
        self._grid = grid
        self._gridworld = VGroup(grid)
        self._gridlines = VGroup(grid)

        if draw_grid:
            self._construct_grid(grid, gridw, animate_grid)
        else:
            if animate_grid:
                self.play(Create(grid))
            else:
                self.add(grid)

        edges: List[Line] = []
        self._edges = VGroup()
        for edge in gridw.edges():
            l = Line(
                self._scale_coordinates(grid, edge.start, grid_size),
                self._scale_coordinates(grid, edge.end, grid_size),
            )
            l.set_stroke(PURPLE)
            edges.append(l)

        self._gridworld.add(*edges)
        self._edges.add(*edges)

        if animate_grid_world:
            self.play(*[Create(edge) for edge in edges])
        else:
            self.add(*edges)

        arrows: List[Arrow] = []
        self._arrows = VGroup()
        for p in gridw.polygons():
            if not p.actions:
                continue

            start_pt = self._scale_coordinates(grid, p.middle_of_polygon(), grid_size)

            for dir in DirectionSets[p.actions]:
                if dir == Direction.L:
                    end_pt = start_pt + np.array([-arrow_length, 0, 0])
                elif dir == Direction.R:
                    end_pt = start_pt + np.array([arrow_length, 0, 0])
                elif dir == Direction.U:
                    end_pt = start_pt + np.array([0, arrow_length, 0])
                elif dir == Direction.D:
                    end_pt = start_pt + np.array([0, -arrow_length, 0])

                arrows.append(
                    Arrow(start_pt, end_pt).set_stroke(GREEN, width=2).set_fill(GREEN)
                )

        self._gridworld.add(*arrows)
        self._arrows.add(*arrows)
        if animate_grid_world:
            self.play(*[Create(arrow) for arrow in arrows])
        else:
            self.add(*arrows)

        target = Polygon(
            *[
                self._scale_coordinates(grid, v, grid_size)
                for v in gridw.target.vertices_in_polygon()
            ]
        )
        target.set_stroke(opacity=0)
        self.add(target)
        self._target = target
        self._gridworld.add(target)

        entry_pt_mob = (
            Dot(self._scale_coordinates(grid, entry_point, grid_size))
            .set_fill(YELLOW)
            .scale(0.5)
        )
        self._entry_pt_mob = entry_pt_mob
        self._gridworld.add(entry_pt_mob)

        if animate_grid_world:
            self.play(
                target.animate.set_fill(ORANGE, opacity=0.5), FadeIn(entry_pt_mob)
            )
            self.wait(2)
        else:
            self.add(entry_pt_mob)
            target.set_fill(ORANGE, opacity=0.5)
