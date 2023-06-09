from src.backward_reachability import BackwardReachabilityTree
from src.polygons import *
from src.linpreds import *

grid_size = 50
gridw_sampler = LinearPredicatesGridWorldSampler(predicate_grid_size=grid_size)
gridw = gridw_sampler.sample(n_preds=7)
gridw.to_prism("grid.prism", grid_size=grid_size)
gridw.draw(grid_size=grid_size)

polygonw = polygons_from_linpreds(gridw)
polygonw.draw()

btree = BackwardReachabilityTree(polygonw)
btree.construct_tree()
btree.draw()
