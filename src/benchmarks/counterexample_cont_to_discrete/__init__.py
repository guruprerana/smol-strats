import json
from gmpy2 import mpq

from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.ordered_edges import OrderedEdgePolicy
from src.polygons.prism import polygon_grid_to_prism
from src.linpreds import Direction, actions_from_directions
from src.polygons import Vertex, simple_rectangular_polygon_gridw

gridw = simple_rectangular_polygon_gridw(size=3)

gridw.root.next.next.add_vertex(Vertex(1, 3)).add_vertex(Vertex(2, 3))
gridw.root = gridw.root.add_vertex(Vertex(mpq(3, 2), 0))
gridw.root.add_diagonal(Vertex(2, 3))
gridw.root.add_diagonal(Vertex(1, 3))
gridw.root.assign_action_polygon(actions_from_directions((Direction.R,)))
gridw.target = gridw.root.next.opp.next.opp
