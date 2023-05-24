from src.linpreds import *

gridw_sampler = LinearPredicatesGridWorldSampler(predicate_grid_size=1000)
gridw = gridw_sampler.sample(n_preds=4)
gridw.to_prism("grid.prism", grid_size=1000)
