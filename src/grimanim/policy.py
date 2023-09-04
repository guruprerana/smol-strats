from collections import defaultdict
from typing import Dict
from manim import *
from src.backward_reachability import VertexInterval
from src.policy.subgoals import SubgoalPolicy, str_nav_dir
from src.game.continuous import ContinuousReachabilityGridGame
from src.polygons import HalfEdge, PolygonGridWorld, Vertex

from src.grimanim.gridworld import Grid
from src.grimanim.colors import *


class Policy(Grid):
    def _construct_policy(
        self,
        gridw: PolygonGridWorld,
        game: ContinuousReachabilityGridGame,
        policy: SubgoalPolicy,
        draw_grid=False,
        animate_grid=False,
        animate_grid_world=False,
        animate_path=True,
        max_iter=5,
    ):
        self._construct_grid_world(
            gridw,
            draw_grid=draw_grid,
            animate_grid=animate_grid,
            animate_grid_world=animate_grid_world,
            entry_point=game.start_coord,
        )
        self.play(self._gridworld.animate.move_to(np.array([-3.5, 0, 0])))

        self._game = game
        self._policy = policy
        game.restart()
        policy.restart()

        self._subgoals_column = VGroup()
        self._subgoals_column.add(Text("Subgoals", color=RED).scale(0.5))
        for subgoal in policy.subgoals:
            text = Text(f"[{subgoal.start}, {subgoal.end}]", color=WHITE).scale(0.3)
            self._subgoals_column.add(text)

        self._subgoals_column.arrange(DOWN).next_to(self._gridworld, RIGHT, buff=0.5)
        self.play(Create(self._subgoals_column))

        # path = self._edge_to_line(game.traversed_edges[0])
        path = VGroup()
        target_reached = False
        self._current_subgoal_group = None
        self._current_subgoal_actions = dict()

        for _ in range(max_iter):
            if target_reached:
                break

            prev_subgoal = policy.current_subgoal
            current_edge = game.current_edge
            dir_e, target_e = policy.navigate(game.current_coord, game.current_edge)
            new_subgoal = policy.current_subgoal
            _, _, target_reached = game.step(dir_e, target_e)

            if new_subgoal != prev_subgoal:
                self._change_subgoal(
                    prev_subgoal, new_subgoal, policy.subgoals[new_subgoal]
                )

            edge_line = self._edge_to_line(current_edge).set_color(GREEN)
            self.add(edge_line)
            self._current_subgoal_actions.get(current_edge, Line()).set_color(GREEN)

            line = self._edge_to_line(game.traversed_edges[-1]).set_color(BLUE)
            self.play(Create(line))
            path.add(line)

            self.remove(edge_line)
            self._current_subgoal_actions.get(current_edge, Line()).set_color(WHITE)

    def _change_subgoal(
        self, prev_subgoal: int, new_subgoal: int, subgoal: VertexInterval
    ) -> None:
        subgoal_line = self._edge_to_line(subgoal).set_color(YELLOW)
        animations = [
            self._subgoals_column.submobjects[1 + prev_subgoal].animate.set_color(GRAY),
            self._subgoals_column.submobjects[1 + new_subgoal].animate.set_color(
                YELLOW
            ),
            Circumscribe(self._subgoals_column.submobjects[1 + new_subgoal]),
            ShowCreationThenFadeOut(subgoal_line),
        ]
        if self._current_subgoal_group:
            animations.append(FadeOut(self._current_subgoal_group))
        self.play(*animations)

        self._current_subgoal_group = VGroup()
        self._current_subgoal_actions: Dict[HalfEdge, Mobject] = dict()

        self._current_subgoal_group.add(
            Text(f"Subgoal [{subgoal.start}, {subgoal.end}]", color=YELLOW).scale(0.4)
        )
        for he in self._policy.edge_actions:
            if len(self._policy.edge_actions[he].targets.get(subgoal, [])) > 0:
                edge_action_str = f"\t[{he.start}, {he.end}] ->\n"
                for vi, dir, e in self._policy.edge_actions[he].targets[subgoal]:
                    edge_action_str += (
                        f"\t\t[{vi.start}, {vi.end}], Preference: {str_nav_dir(dir)}\n"
                    )

                edge_action_text = Text(
                    edge_action_str, color=WHITE, font_size=12, line_spacing=1
                )
                self._current_subgoal_actions[he] = edge_action_text
                self._current_subgoal_group.add(edge_action_text)

        self._current_subgoal_group.arrange(DOWN).next_to(
            self._subgoals_column, RIGHT, buff=0.5
        )
        self.play(FadeIn(self._current_subgoal_group))
