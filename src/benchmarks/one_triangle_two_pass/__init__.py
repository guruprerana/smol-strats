import json

from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy
from src.polygons.prism import polygon_grid_to_prism
from src.linpreds import Direction, actions_from_directions
from src.polygons import Vertex, simple_rectangular_polygon_gridw

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


def main():
    gridw.draw("benchmarks/one_triangle_two_pass/polygon-grid.png")

    policy = SubgoalPolicy(gridw)
    policy.build(btree_depth=10)

    game = ContinuousReachabilityGridGame(
        gridw, policy.start_leaf.linked_edge, Vertex(0, 0)
    )
    success = game.run(policy)
    print(f"{success}-ly won the game")

    btree = policy.btree
    btree.draw(filename="benchmarks/one_triangle_two_pass/backward-graph.png")
    game.draw(filename="benchmarks/one_triangle_two_pass/subgoals-policy-path.png")

    polygon_grid_to_prism(
        gridw, "benchmarks/one_triangle_two_pass/one_triangle_two_pass.prism", 100
    )

    # game = GymGame(gridw, gridw.root)
    # game.reset()
    # game.step([0])
    # game.step([0.49])
    # _, _, terminated, _, _ = game.step([1])
    # if terminated:
    #     print(f"Gym game won")

    policy.to_pseudocode(
        "benchmarks/one_triangle_two_pass/subgoal_policy_pseudocode.txt"
    )

    policy_serialized = SubgoalPolicySerializer.serialize(policy)
    with open("benchmarks/one_triangle_two_pass/polygongrid.json", "w") as f:
        json.dump(policy_serialized[0], f)

    with open("benchmarks/one_triangle_two_pass/subgoals_policy.json", "w") as f:
        json.dump(policy_serialized, f)

    with open("benchmarks/one_triangle_two_pass/subgoals_policy.json", "r") as f:
        policy_json = json.load(f)
        policy = SubgoalPolicySerializer.deserialize(policy_json)

    game = ContinuousReachabilityGridGame(
        policy.gridw, policy.start_edge, policy.start_point
    )
    success = game.run(policy)
    print(f"{success}-ly won the game again!")

    return gridw, btree
