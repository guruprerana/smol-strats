import random
from typing import Optional, Tuple, Union

Point = Tuple[int, int]


def _p_diff(p1: Point, p2: Point) -> Point:
    return Point(p1[0] - p2[0], p1[1] - p2[1])


def _scale_point(p: Point, factor: int) -> Point:
    return Point(factor * p[0], factor * p[1])


class LinearPredicate:
    """
    Represents a linear predicate as a line joining two points on the boundary of a square grid
    """

    def __init__(self, p1: Point, p2: Point, predicate_grid_size=1000) -> None:
        """Initializes a linear predicate with two specified points

        Args:
            p1 (Point): first input point
            p2 (Point): second input point
            predicate_grid_size (int, optional): size of the grid on which the input points are considered. Defaults to 1000.
        """
        self.p1: Point = p1
        self.p2: Point = p2
        self.predicate_grid_size = predicate_grid_size

    def determine_side(self, p: Point, grid_size: Optional[int]) -> bool:
        if not grid_size:
            grid_size = self.predicate_grid_size

        scale_factor = grid_size // self.predicate_grid_size
        scaled_p1, scaled_p2 = _scale_point(self.p1, scale_factor), _scale_point(
            self.p2, scale_factor
        )
        (v11, v12), (v21, v22) = _p_diff(scaled_p2, scaled_p1), _p_diff(p, scaled_p1)

        # compute detemrinant of two vectors and decide based on >= 0
        return ((v11 * v22) - (v12 * v21)) >= 0


class LinearPredicateSampler:
    def __init__(self, predicate_grid_size=1000) -> None:
        random.seed()
        self.predicate_grid_size = predicate_grid_size

    def sample(self) -> LinearPredicate:
        # first sample the two sides of the square which we want the points to be on
        sides = random.sample((0, 1, 2, 3), k=2)
        # sample two points along the two sides
        p1 = random.randrange(0, self.predicate_grid_size)
        p2 = random.randrange(0, self.predicate_grid_size)

        # convert to points
        p1 = self._convert_side_to_point(sides[0], p1)
        p2 = self._convert_side_to_point(sides[1], p2)

        return LinearPredicate(p1, p2, self.predicate_grid_size)

    def _convert_side_to_point(self, side: int, p: int) -> Point:
        if not (0 <= p < side):
            raise ValueError

        if side == 0:
            return Point(p, 0)
        elif side == 1:
            return Point(self.predicate_grid_size, p)
        elif side == 2:
            return Point(1 + p, self.predicate_grid_size)
        elif side == 3:
            return Point(0, 1 + p)
        else:
            raise ValueError
