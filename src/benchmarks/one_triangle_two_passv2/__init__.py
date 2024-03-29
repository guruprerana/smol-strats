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

gridw.root.navigate("non").add_vertex(Vertex(25, 30)).add_vertex(Vertex(25, 20))
gridw.root.navigate("nonnno").add_diagonal(Vertex(50, 50))
gridw.root.navigate("nonno").add_diagonal(Vertex(50, 0))

gridw.root.navigate("nono").assign_action_polygon(
    actions_from_directions([Direction.R, Direction.L])
)
gridw.root.navigate("nonno").assign_action_polygon(
    actions_from_directions([Direction.U, Direction.D])
)
gridw.root.navigate("nonnno").assign_action_polygon(
    actions_from_directions([Direction.L, Direction.R])
)

policy = SubgoalPolicy(gridw, start_point=Vertex(0, 10))
policy.build(btree_depth=10)

game = ContinuousReachabilityGridGame(
    gridw, policy.start_leaf.linked_edge, Vertex(0, 10)
)
success = game.run(policy)


def main():
    global gridw, game, policy, success
    gridw.draw(
        "benchmarks/one_triangle_two_passv2/polygon-grid.png", 
        start_point=Vertex(0, 10), 
        edge_width=7, 
        start_point_radius=10,
    )

    print(f"{success}-ly won the game")

    btree = policy.btree
    btree.draw(
        filename="benchmarks/one_triangle_two_passv2/backward-graph.png",
        start_point=Vertex(0, 10),
    )
    game.draw(
        filename="benchmarks/one_triangle_two_passv2/subgoals-policy-path.png",
        start_point=Vertex(0, 10),
        edge_width=7,
        start_point_radius=10,
    )

    polygon_grid_to_prism(
        gridw,
        "benchmarks/one_triangle_two_passv2/one_triangle_two_passv2.prism",
        100,
        start_pt=Vertex(0, 10),
    )

    # game = GymGame(gridw, gridw.root)
    # game.reset()
    # game.step([0])
    # game.step([0.49])
    # _, _, terminated, _, _ = game.step([1])
    # if terminated:
    #     print(f"Gym game won")

    policy.to_pseudocode(
        "benchmarks/one_triangle_two_passv2/subgoal_policy_pseudocode.txt"
    )

    policy_serialized = SubgoalPolicySerializer.serialize(policy)
    with open("benchmarks/one_triangle_two_passv2/polygongrid.json", "w") as f:
        json.dump((policy_serialized[0], policy_serialized[-1]), f)

    with open("benchmarks/one_triangle_two_passv2/subgoals_policy.json", "w") as f:
        json.dump(policy_serialized, f)

    with open("benchmarks/one_triangle_two_passv2/subgoals_policy.json", "r") as f:
        policy_json = json.load(f)
        policy = SubgoalPolicySerializer.deserialize(policy_json)

    game = ContinuousReachabilityGridGame(
        policy.gridw, policy.start_edge, policy.start_point
    )
    success = game.run(policy)
    print(f"{success}-ly won the game again!")

    return gridw, btree, policy, game
