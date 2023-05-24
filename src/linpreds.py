from enum import Enum
from itertools import chain, combinations
import random
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

Point = Tuple[int, int]


def _p_diff(p1: Point, p2: Point) -> Point:
    return (p1[0] - p2[0], p1[1] - p2[1])


def _scale_point(p: Point, factor: int) -> Point:
    return (factor * p[0], factor * p[1])


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

        self._scale_cache: Dict[int, Tuple[Point, Point]] = {}

    def determine_side(self, p: Point, grid_size: Optional[int]) -> bool:
        scaled_p1, scaled_p2 = self._scale_points(grid_size)
        (v11, v12), (v21, v22) = _p_diff(p, scaled_p1), _p_diff(scaled_p2, scaled_p1)

        # compute detemrinant of two vectors and decide based on >= 0
        return ((v11 * v22) - (v12 * v21)) >= 0

    def predicate_string(self, pos: bool, grid_size: Optional[int]) -> str:
        scaled_p1, scaled_p2 = self._scale_points(grid_size)
        fx, fy = _p_diff(scaled_p2, scaled_p1)

        op = ">=" if pos else "<"

        return f"{fy}*(x - {scaled_p1[0]}) - {fx}*(y - {scaled_p1[1]}) {op} 0"

    def _scale_points(self, grid_size: Optional[int]) -> Tuple[Point, Point]:
        if not grid_size:
            grid_size = self.predicate_grid_size

        scale_factor = grid_size // self.predicate_grid_size

        if scale_factor in self._scale_cache:
            return self._scale_cache[scale_factor]

        scaled_p1, scaled_p2 = _scale_point(self.p1, scale_factor), _scale_point(
            self.p2, scale_factor
        )
        self._scale_cache[scale_factor] = scaled_p1, scaled_p2
        return scaled_p1, scaled_p2


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
        assert 0 <= p < self.predicate_grid_size

        if side == 0:
            return (p, 0)
        elif side == 1:
            return (self.predicate_grid_size, p)
        elif side == 2:
            return (1 + p, self.predicate_grid_size)
        elif side == 3:
            return (0, 1 + p)
        else:
            raise ValueError


T = TypeVar("T")


# taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable: Iterable[T]) -> Tuple[Tuple[T, ...], ...]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


Direction = Enum("Direction", ["L", "R", "U", "D"])
DirectionSets = list(powerset((Direction.L, Direction.R, Direction.U, Direction.D)))


class LinearPredicatesGridWorld:
    def __init__(self, preds: List[LinearPredicate], actions: List[int]) -> None:
        assert len(preds) > 0
        assert all(
            preds[i].predicate_grid_size == preds[0].predicate_grid_size
            for i in range(1, len(preds))
        )
        assert len(actions) == 1 << len(preds)
        assert all(0 <= action < 1 << 4 for action in actions)

        self.preds = preds
        self.predicate_grid_size = preds[0].predicate_grid_size
        self.actions = actions
        self._n_preds = len(preds)

    def _region_prism_predicate(self, region: int, grid_size: int) -> str:
        assert 0 <= region < len(self.actions)

        pred_strings: List[str] = []
        for i, p in enumerate(bin(region)[2:].zfill(self._n_preds)):
            pred_strings.append(
                self.preds[i].predicate_string(p == "1", grid_size=grid_size)
            )

        return " & ".join(pred_strings)

    def _prism_of_region(self, region: int, grid_size: int) -> List[str]:
        pred_string = self._region_prism_predicate(region, grid_size)

        action_strings: List[str] = []
        for dir in DirectionSets[self.actions[region]]:
            if dir == Direction.L:
                action_strings.append(f"[] {pred_string} -> (x' = x - 1);")
            elif dir == Direction.R:
                action_strings.append(f"[] {pred_string} -> (x' = x + 1);")
            elif dir == Direction.U:
                action_strings.append(f"[] {pred_string} -> (y' = y + 1);")
            elif dir == Direction.D:
                action_strings.append(f"[] {pred_string} -> (y' = y - 1);")
            else:
                raise ValueError

        return action_strings

    def _region_of_point(self, p: Point, grid_size: Optional[int]) -> int:
        if not grid_size:
            grid_size = self.predicate_grid_size

        sides = [pred.determine_side(p, grid_size) for pred in self.preds]
        return sum(1 << i for i in range(self._n_preds) if sides[i])

    def to_prism(self, filename: str, grid_size=1000) -> None:
        with open(filename, "w") as f:
            f.writelines(
                (
                    "mdp\n",
                    "\n",
                    "module grid\n",
                    f"\tx : [0..{grid_size}] init 0;\n",
                    f"\ty : [0..{grid_size}] init 0;\n",
                    "\n",
                )
            )

            for i in range(1 << self._n_preds):
                action_strings = self._prism_of_region(i, grid_size)
                action_strings.append("\n")
                action_strings = [f"\t{action}\n" for action in action_strings]

                f.writelines(action_strings)

            target_pt = grid_size, grid_size
            target_region = self._region_of_point(target_pt, grid_size)
            region_pred = self._region_prism_predicate(target_region, grid_size)

            f.writelines("endmodule\n\n")

            f.writelines((f'label "target" = {region_pred};\n\n',))


class LinearPredicatesGridWorldSampler:
    def __init__(self, predicate_grid_size=1000) -> None:
        self.predicate_grid_size = predicate_grid_size

    def sample(self, n_preds=7) -> LinearPredicatesGridWorld:
        pred_sampler = LinearPredicateSampler(
            predicate_grid_size=self.predicate_grid_size
        )

        preds = [pred_sampler.sample() for _ in range(n_preds)]
        actions = [random.randrange(0, 1 << 4) for _ in range(1 << n_preds)]

        return LinearPredicatesGridWorld(preds, actions)
