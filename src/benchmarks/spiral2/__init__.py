import json
from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.game.continuous import ContinuousReachabilityGridGame
from src.polygons.prism import polygon_grid_to_prism
from src.linpreds import Direction, actions_from_directions
from src.polygons import HalfEdge, Vertex, simple_rectangular_polygon_gridw

gridw = simple_rectangular_polygon_gridw(size=28)
gridw.root.prev.add_diagonal(Vertex(28, 28))
gridw.root.prev.opp.add_vertex(Vertex(14, 14)).add_vertex(Vertex(13, 13))
gridw.root.prev.opp.prev.prev.add_diagonal(Vertex(14, 14))
gridw.root = gridw.root.add_vertex(Vertex(26, 0))
gridw.root.add_diagonal(Vertex(13, 13))
gridw.root.next.add_vertex(Vertex(14, 12))
gridw.root.next.next.opp.add_diagonal(Vertex(14, 14))

gridw.root.assign_action_polygon(actions_from_directions([Direction.D, Direction.R]))
gridw.root.next.opp.assign_action_polygon(actions_from_directions([Direction.U, Direction.R]))
gridw.root.prev.opp.assign_action_polygon(actions_from_directions([Direction.D, Direction.L]))
gridw.root.prev.opp.next.next.opp.assign_action_polygon(actions_from_directions([Direction.U, Direction.L]))

gridw.target = gridw.root.next.next.opp

policy = SubgoalPolicy(gridw)
policy.build(btree_depth=1000)

game = ContinuousReachabilityGridGame(
    gridw, policy.start_leaf.linked_edge, Vertex(0, 0)
)
success = game.run(policy)


def main():
    global gridw, game, policy, success
    gridw.draw("benchmarks/spiral2/polygon-grid.png", edge_width=5, start_point_radius=10)

    print(f"{success}-ly won the game")

    btree = policy.btree
    btree.draw(filename="benchmarks/spiral2/backward-graph.png")
    game.draw(filename="benchmarks/spiral2/subgoals-policy-path.png", edge_width=5, start_point_radius=10)

    polygon_grid_to_prism(gridw, "benchmarks/spiral2/spiral.prism", 100)

    policy.to_pseudocode("benchmarks/spiral2/subgoal_policy_pseudocode.txt")

    policy_serialized = SubgoalPolicySerializer.serialize(policy)
    with open("benchmarks/spiral2/polygongrid.json", "w") as f:
        json.dump((policy_serialized[0], policy_serialized[-1]), f)

    with open("benchmarks/spiral2/subgoals_policy.json", "w") as f:
        json.dump(policy_serialized, f)

    with open("benchmarks/spiral2/subgoals_policy.json", "r") as f:
        policy_json = json.load(f)
        policy = SubgoalPolicySerializer.deserialize(policy_json)

    game = ContinuousReachabilityGridGame(
        policy.gridw, policy.start_edge, policy.start_point
    )
    success = game.run(policy)
    print(f"{success}-ly won the game again!")

    return gridw, btree, policy, game
