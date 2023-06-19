from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy
from src.backward_reachability import BackwardReachabilityTree
from src.polygons import *
from src.linpreds import *

grid_size = 50
gridw_sampler = LinearPredicatesGridWorldSampler(predicate_grid_size=grid_size)
gridw = gridw_sampler.sample(n_preds=7)
gridw.to_prism("benchmarks/generated/generated.prism", grid_size=grid_size)
gridw.draw(grid_size=grid_size, filename="benchmarks/generated/grid.png")

polygonw = polygons_from_linpreds(gridw)
polygonw.draw(filename="benchmarks/generated/polygon-grid.png")

policy = OrderedEdgePolicy(polygonw)
policy.build(btree_depth=100)

game = ContinuousReachabilityGridGame(
    polygonw, policy.start_leaf.linked_edge, Vertex(0, 0)
)
success = game.run(policy)
print(f"{success}-ly won the game")

btree = policy.btree
btree.draw(filename="benchmarks/generated/backward-graph.png")
game.draw(filename="benchmarks/generated/ordered-edges-policy-path.png")
