import unittest

from src.linpreds import (
    LinearPredicate,
    LinearPredicatesGridWorld,
    LinearPredicatesGridWorldSampler,
)
from src.polygons import HalfEdge, polygons_from_linpreds


class TestPolygonGridWorld(unittest.TestCase):
    def test_polygons_from_linpreds1(self):
        preds = [
            LinearPredicate((100, 0), (500, 1000)),
            LinearPredicate((0, 200), (1000, 200)),
        ]
        actions = [13, 12, 11, 10]
        lingrid = LinearPredicatesGridWorld(preds, actions)
        polygons = polygons_from_linpreds(lingrid)

        # def print_edge(e: HalfEdge) -> None:
        #     # print(repr(e))
        #     print(e.actions)

        # polygons.traverse(print_edge)
        # print(id(polygons.root.next.opp) == id(polygons.root.next.opp.prev.next))

    def test_polygons_from_linpreds2(self):
        grid_size = 1000
        gridw_sampler = LinearPredicatesGridWorldSampler(predicate_grid_size=grid_size)
        gridw = gridw_sampler.sample(n_preds=20)
        polygons = polygons_from_linpreds(gridw)


if __name__ == "__main__":
    unittest.main()
