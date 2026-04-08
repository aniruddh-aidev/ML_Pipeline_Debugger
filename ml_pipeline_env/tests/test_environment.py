"""
Integration tests for MLPipelineEnvironment.
Run: pytest ml_pipeline_env/tests/test_environment.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.ml_pipeline_environment import (
    MLPipelineEnvironment, TASK_ORDER, MAX_STEPS_PER_TASK,
)
from ml_pipeline_env.models import MLPipelineAction
from ml_pipeline_env.tasks import TASKS


def perfect_action(task_id: str) -> MLPipelineAction:
    return MLPipelineAction(fix=TASKS[task_id].fixed_code)

def bad_action() -> MLPipelineAction:
    return MLPipelineAction(fix="# no fix here")


class TestReset:

    def test_reset_returns_first_task(self):
        env = MLPipelineEnvironment()
        obs = env.reset()
        assert obs.task_id == TASK_ORDER[0]

    def test_reset_done_is_false(self):
        env = MLPipelineEnvironment()
        obs = env.reset()
        assert obs.done is False

    def test_reset_step_count_zero(self):
        env = MLPipelineEnvironment()
        obs = env.reset()
        assert obs.step_count == 0

    def test_reset_no_hint_initially(self):
        env = MLPipelineEnvironment()
        obs = env.reset()
        assert obs.hint is None

    def test_double_reset_restarts(self):
        env = MLPipelineEnvironment()
        env.reset()
        env.step(bad_action())
        obs = env.reset()
        assert obs.task_id == TASK_ORDER[0]
        assert obs.step_count == 0


class TestStep:

    def test_bad_action_increments_step(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = env.step(bad_action())
        assert obs.step_count == 1

    def test_bad_action_shows_hint_after_first_fail(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = env.step(bad_action())
        assert obs.hint is not None

    def test_bad_action_score_zero(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = env.step(bad_action())
        assert obs.score == 0.0

    def test_perfect_action_advances_task(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = env.step(perfect_action(TASK_ORDER[0]))
        assert obs.task_id == TASK_ORDER[1]

    def test_solve_all_tasks_marks_done(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = None
        for task_id in TASK_ORDER:
            obs = env.step(perfect_action(task_id))
        assert obs.done is True

    def test_exhausted_attempts_advances_task(self):
        env = MLPipelineEnvironment()
        env.reset()
        for _ in range(MAX_STEPS_PER_TASK):
            obs = env.step(bad_action())
        assert obs.task_id == TASK_ORDER[1]

    def test_step_after_done_returns_done(self):
        env = MLPipelineEnvironment()
        env.reset()
        for task_id in TASK_ORDER:
            env.step(perfect_action(task_id))
        obs = env.step(bad_action())
        assert obs.done is True

    def test_state_step_count_matches(self):
        env = MLPipelineEnvironment()
        env.reset()
        env.step(bad_action())
        env.step(bad_action())
        assert env.state.step_count == 2


class TestFullEpisode:

    def test_full_perfect_episode(self):
        env = MLPipelineEnvironment()
        env.reset()
        final_obs = None
        for task_id in TASK_ORDER:
            final_obs = env.step(perfect_action(task_id))
        assert final_obs.done is True

    def test_full_bad_episode_terminates(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = None
        for _ in range(MAX_STEPS_PER_TASK * len(TASK_ORDER) + 2):
            obs = env.step(bad_action())
            if obs.done:
                break
        assert obs.done is True

    def test_mixed_episode(self):
        env = MLPipelineEnvironment()
        env.reset()
        obs = env.step(perfect_action("task_easy"))
        assert obs.task_id == "task_medium"
        for _ in range(MAX_STEPS_PER_TASK):
            obs = env.step(bad_action())
        assert obs.task_id == "task_hard"
        obs = env.step(perfect_action("task_hard"))
        assert obs.done is True
