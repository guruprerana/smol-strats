from src.backward_reachability import BackwardReachabilityTree
from src.linpreds import Direction, actions_from_directions
from src.polygons import Vertex, simple_rectangular_polygon_gridw


def main():
    gridw = simple_rectangular_polygon_gridw(size=50)

    gridw.root.assign_action_polygon(actions_from_directions((Direction.R,)))
    gridw.root.next.opp.assign_action_polygon(
        actions_from_directions((Direction.L, Direction.R))
    )
    gridw.root.next.opp.next.opp.assign_action_polygon(
        actions_from_directions((Direction.U, Direction.L))
    )
    gridw.root.next.opp.prev.opp.assign_action_polygon(
        actions_from_directions((Direction.L,))
    )
    gridw.target = gridw.root.next.opp.prev.opp
    gridw.draw("src/benchmarks/one_triangle_two_pass/polygon-grid.png")

    print(gridw.root.next.opp.next)

    btree = BackwardReachabilityTree(gridw)
    btree.construct_tree(max_depth=8)
    btree.print()
    btree.draw(filename="src/benchmarks/one_triangle_two_pass/backward-graph.png")
