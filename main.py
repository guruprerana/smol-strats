import json
import pickle
from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy, OrderedEdgePolicySerializer
from src.backward_reachability import BackwardReachabilityTree
from src.polygons import *
from src.linpreds import *

grid_size = 2
gridw_sampler = LinearPredicatesGridWorldSampler(predicate_grid_size=grid_size)
gridw = gridw_sampler.sample(n_preds=2)
gridw.to_prism("benchmarks/generated/generated.prism", grid_size=grid_size)
gridw.draw(grid_size=grid_size, filename="benchmarks/generated/grid.png")

print(gridw.preds)

polygonw = polygons_from_linpreds(gridw)

btree = BackwardReachabilityTree(polygonw, None)
btree.construct_tree(max_depth=1000)

start_leaf = btree.max_depth_leaf()
start_edge, start_point = start_leaf.linked_edge, start_leaf.edge.start

policy = OrderedEdgePolicy(
    polygonw,
    start_edge=start_edge,
    start_point=start_point,
    btree=btree,
    start_leaf=start_leaf,
)
policy.build()

game = ContinuousReachabilityGridGame(polygonw, start_edge, start_point)
success = game.run(policy)
print(f"{success}-ly won the game")

btree = policy.btree
polygonw.draw(filename="benchmarks/generated/polygon-grid.png", start_point=start_point)
btree.draw(filename="benchmarks/generated/backward-graph.png", start_point=start_point)
game.draw(
    filename="benchmarks/generated/ordered-edges-policy-path.png",
    start_point=start_point,
)

policy.restart()
game.restart()

with open("benchmarks/generated/pickles/linpreds.pickle", "wb") as f:
    pickle.dump(gridw, f)

with open("benchmarks/generated/pickles/polygonw.pickle", "wb") as f:
    pickle.dump(polygonw, f)

with open("benchmarks/generated/pickles/policy.pickle", "wb") as f:
    pickle.dump(policy, f)

with open("benchmarks/generated/pickles/game.pickle", "wb") as f:
    pickle.dump(game, f)

with open("benchmarks/generated/ordered_edges_policy.json", "w") as f:
    json.dump(OrderedEdgePolicySerializer.serialize(policy), f)
