import json

from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy
from src.polygons.prism import polygon_grid_to_prism
from src.linpreds import Direction, actions_from_directions
from src.polygons import Vertex, simple_rectangular_polygon_gridw

def main():
    with open("benchmarks/size3preds5loopy/subgoals_policy.json", "r") as f:
        policy_json = json.load(f)
        policy = SubgoalPolicySerializer.deserialize(policy_json)

    game = ContinuousReachabilityGridGame(
        policy.gridw, policy.start_edge, policy.start_point
    )
    success = game.run(policy)

    policy.gridw.draw(
        "benchmarks/size3preds5loopy/polygon-grid.png",
        scale=200,
        edge_width=7, 
        start_point_radius=10,
    )
    game.draw(
        filename="benchmarks/size3preds5loopy/subgoals-policy-path.png",
        scale=200,
        edge_width=5,
        start_point_radius=7,
    )

    print(f"{success}-ly won the game again!")
