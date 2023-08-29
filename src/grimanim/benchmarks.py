from src.grimanim.policy import Policy
from src.benchmarks.one_triangle_two_passv2 import gridw, policy, game
from src.benchmarks.spiral import (
    gridw as spiral_gridw,
    policy as spiral_policy,
    game as spiral_game,
)
from src.grimanim.path import PolicyPath
from src.grimanim.btree_with_tree import BTree


class OneTriangleTwoPassV2(PolicyPath):
    def construct(self):
        self._construct_policy(gridw, game)
        self.wait(2)


class Spiral(PolicyPath):
    def construct(self):
        self._construct_policy(spiral_gridw, spiral_game)
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
        return super()._construct_policy(
            spiral_gridw, spiral_game, spiral_policy, max_iter=30
        )
