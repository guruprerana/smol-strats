from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy
from src.polygons.prism import polygon_grid_to_prism
from src.backward_reachability import BackwardReachabilityTree
from src.linpreds import Direction, actions_from_directions
from src.polygons import Vertex, simple_rectangular_polygon_gridw


def main():
    gridw = simple_rectangular_polygon_gridw(size=50)

    gridw.root.prev.prev.add_vertex(Vertex(25, 50))
    gridw.root.prev.add_vertex(Vertex(0, 25))
    gridw.root = gridw.root.add_vertex(Vertex(25, 0))
    # gridw.traverse(lambda edge: print(str(edge)))
    gridw.root.add_diagonal(Vertex(25, 50))
    gridw.root.next.add_diagonal(Vertex(0, 25))
    gridw.root.next.next.add_diagonal(Vertex(25, 0))

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
    gridw.draw("benchmarks/one_triangle_two_pass/polygon-grid.png")

    policy = OrderedEdgePolicy(gridw)
    policy.build(btree_depth=10)

    game = ContinuousReachabilityGridGame(
        gridw, policy.start_leaf.linked_edge, Vertex(0, 0)
    )
    success = game.run(policy)
    print(f"{success}-ly won the game")

    btree = policy.btree
    btree.draw(filename="benchmarks/one_triangle_two_pass/backward-graph.png")
    game.draw(filename="benchmarks/one_triangle_two_pass/ordered-edges-policy-path.png")

    polygon_grid_to_prism(
        gridw, "benchmarks/one_triangle_two_pass/one_triangle_two_pass.prism", 100
    )

    return gridw, btree
