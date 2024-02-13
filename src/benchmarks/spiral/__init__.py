import json
from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.game.continuous import ContinuousReachabilityGridGame
from src.polygons.prism import polygon_grid_to_prism
from src.linpreds import Direction, actions_from_directions
from src.polygons import HalfEdge, Vertex, simple_rectangular_polygon_gridw

gridw = simple_rectangular_polygon_gridw(size=27)
gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 21), Vertex(27, 21)))
gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 11), Vertex(27, 11)))
gridw.root.prev.add_intersection_edge(HalfEdge(Vertex(0, 10), Vertex(27, 10)))
gridw.root.navigate("nfn").add_vertex(Vertex(10, 11)).add_vertex(
    Vertex(13, 11)
).add_vertex(Vertex(14, 11)).add_vertex(Vertex(17, 11))
gridw.root.navigate("nn").add_vertex(Vertex(10, 10)).add_vertex(
    Vertex(13, 10)
).add_vertex(Vertex(14, 10)).add_vertex(Vertex(17, 10))
gridw.root = gridw.root.add_vertex(Vertex(24, 0)).add_vertex(Vertex(23, 0))

gridw.root.navigate("pbbp").add_diagonal(Vertex(10, 11))
gridw.root.navigate("pbb").assign_action_polygon(
    actions_from_directions([Direction.D, Direction.L])
)
gridw.root.navigate("pbbbp").add_diagonal(Vertex(17, 11))
gridw.root.navigate("pbbb").assign_action_polygon(
    actions_from_directions([Direction.L, Direction.U])
)
gridw.root.navigate("pbbbb").assign_action_polygon(
    actions_from_directions([Direction.U, Direction.R])
)
gridw.root.navigate("ppo").add_diagonal(Vertex(10, 11))
gridw.root.navigate("ppo").assign_action_polygon(
    actions_from_directions([Direction.D, Direction.L])
)
gridw.root.prev.add_diagonal(Vertex(10, 10))
gridw.root.prev.opp.assign_action_polygon(
    actions_from_directions([Direction.D, Direction.L])
)
gridw.root.navigate("bpbppop").add_diagonal(Vertex(14, 11))
gridw.root.navigate("bpbo").add_diagonal(Vertex(13, 11))
gridw.target = gridw.root.navigate("pppo")
gridw.root.navigate("ppppo").add_diagonal(Vertex(17, 11))
gridw.root.navigate("ppppof").assign_action_polygon(
    actions_from_directions([Direction.U, Direction.R])
)
gridw.root.navigate("nn").add_diagonal(Vertex(17, 10))
gridw.root.navigate("nnf").assign_action_polygon(
    actions_from_directions([Direction.U, Direction.R])
)
gridw.root.navigate("n").add_diagonal(Vertex(14, 10))
gridw.root.navigate("nf").assign_action_polygon(
    actions_from_directions([Direction.R, Direction.D])
)
gridw.root.add_diagonal(Vertex(13, 10))
gridw.root.navigate("f").assign_action_polygon(
    actions_from_directions([Direction.U, Direction.R])
)
gridw.root.assign_action_polygon(actions_from_directions([Direction.R, Direction.D]))

policy = SubgoalPolicy(gridw)
policy.build(btree_depth=1000)

game = ContinuousReachabilityGridGame(
    gridw, policy.start_leaf.linked_edge, Vertex(0, 0)
)
success = game.run(policy)


def main():
    global gridw, game, policy, success
    gridw.draw("benchmarks/spiral/polygon-grid.png", edge_width=5, start_point_radius=10)

    print(f"{success}-ly won the game")

    btree = policy.btree
    btree.draw(filename="benchmarks/spiral/backward-graph.png")
    game.draw(filename="benchmarks/spiral/subgoals-policy-path.png", edge_width=5, start_point_radius=10)

    polygon_grid_to_prism(gridw, "benchmarks/spiral/spiral.prism", 100)

    policy.to_pseudocode("benchmarks/spiral/subgoal_policy_pseudocode.txt")

    policy_serialized = SubgoalPolicySerializer.serialize(policy)
    with open("benchmarks/spiral/polygongrid.json", "w") as f:
        json.dump((policy_serialized[0], policy_serialized[-1]), f)

    with open("benchmarks/spiral/subgoals_policy.json", "w") as f:
        json.dump(policy_serialized, f)

    with open("benchmarks/spiral/subgoals_policy.json", "r") as f:
        policy_json = json.load(f)
        policy = SubgoalPolicySerializer.deserialize(policy_json)

    game = ContinuousReachabilityGridGame(
        policy.gridw, policy.start_edge, policy.start_point
    )
    success = game.run(policy)
    print(f"{success}-ly won the game again!")

    return gridw, btree, policy, game
