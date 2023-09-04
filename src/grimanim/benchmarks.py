import json
from manim import *
from src.backward_reachability import BackwardReachabilityTree
from src.game.continuous import ContinuousReachabilityGridGame
from src.policy.subgoals import SubgoalPolicy, SubgoalPolicySerializer
from src.polygons import HalfEdge, Vertex

from src.grimanim.policy import Policy
from src.benchmarks.one_triangle_two_passv2 import gridw, policy, game
from src.benchmarks.spiral import (
    gridw as spiral_gridw,
    policy as spiral_policy,
    game as spiral_game,
)
from src.benchmarks.diagonal import (
    gridw as diagonal_gridw,
    policy as diagonal_policy,
    game as diagonal_game,
)
from src.benchmarks.counterexample_cont_to_discrete import gridw as counterexample_gridw
from src.grimanim.gridworld import Grid
from src.grimanim.path import PolicyPath
from src.grimanim.btree_with_tree import BTree


class OneTriangleTwoPassV2Gridworld(Grid):
    def construct(self):
        self._construct_grid_world(gridw, entry_point=policy.start_point)
        self.wait(5)


class DiagonalGridworld(Grid):
    def construct(self):
        self._construct_grid_world(
            diagonal_gridw,
            entry_point=diagonal_policy.start_point,
            animate_grid=False,
            animate_grid_world=False,
        )
        self.play(FadeOut(self._gridlines), run_time=1)
        self.wait(1)


class DiagonalPolicyPathDiscrete(PolicyPath):
    def construct(self):
        traversed_path = []
        for i in range(8):
            traversed_path.append(HalfEdge(Vertex(i, i), Vertex(i, i + 1)))
            traversed_path.append(HalfEdge(Vertex(i, i + 1), Vertex(i + 1, i + 1)))
        traversed_path.append(HalfEdge(Vertex(8, 8), Vertex(8, 9)))
        diagonal_game.traversed_edges = traversed_path
        self._construct_policy(diagonal_gridw, diagonal_game, draw_grid=True)
        self.wait(2)


class DiagonalPolicyPathContinuous(PolicyPath):
    def construct(self):
        self._construct_policy(diagonal_gridw, diagonal_game)
        self.wait(2)


class OneTriangleTwoPassV2PolicyPath(PolicyPath):
    def construct(self):
        self._construct_policy(gridw, game, draw_grid=True)
        self.wait(5)


class SpiralPolicyPath(PolicyPath):
    def construct(self):
        self._construct_policy(
            spiral_gridw,
            spiral_game,
            draw_grid=True,
            animate_grid=True,
            animate_grid_world=True,
        )
        self.wait(2)


class OneTriangleTwoPassV2BTree(BTree):
    def construct(self):
        self._construct_btree(gridw, game, policy.btree, max_depth=2)
        self.wait(2)


class SpiralBTree(BTree):
    def construct(self):
        self._construct_btree(
            spiral_gridw, spiral_game, spiral_policy.btree, max_depth=20
        )


class SpiralPolicy(Policy):
    def construct(self):
        return self._construct_policy(
            spiral_gridw, spiral_game, spiral_policy, max_iter=30
        )


class Size3Preds5LoopyBTree(BTree):
    def construct(self):
        with open("benchmarks/size3preds5loopy/subgoals_policy.json", "r") as f:
            policy = SubgoalPolicySerializer.deserialize(json.load(f))

        btree = BackwardReachabilityTree(policy.gridw, policy.start_point)
        btree.construct_tree(max_depth=10)

        game = ContinuousReachabilityGridGame(
            gridw, policy.start_edge, policy.start_point
        )
        return self._construct_btree(policy.gridw, game, btree, max_depth=2)


class CounterexampleContToDiscrete(Grid):
    def construct(self):
        self._construct_grid_world(counterexample_gridw, entry_point=Vertex(0, 1))
        self.wait(2)
