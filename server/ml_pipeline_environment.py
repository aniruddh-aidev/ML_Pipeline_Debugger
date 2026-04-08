"""
ML Pipeline Debugger — Environment Logic
Extends openenv.core.env_server.Environment.
"""

import uuid
from typing import Optional
from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from ml_pipeline_env.models import MLPipelineAction, MLPipelineObservation
from ml_pipeline_env.tasks import TASKS

TASK_ORDER = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS_PER_TASK = 5


class MLPipelineEnvironment(Environment[MLPipelineAction, MLPipelineObservation, State]):
    """
    An OpenEnv environment where an LLM agent is shown a broken ML pipeline
    and must submit a corrected version. Three tasks of increasing difficulty.
    """

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._task_index: int = 0
        self._task_steps: int = 0
        self._current_task = TASKS[TASK_ORDER[0]]
        self._show_hint: bool = False
        self._done: bool = False

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> MLPipelineObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._task_index = 0
        self._task_steps = 0
        self._current_task = TASKS[TASK_ORDER[0]]
        self._show_hint = False
        self._done = False
        return self._make_observation()

    def step(
        self,
        action: MLPipelineAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> MLPipelineObservation:
        if self._done:
            return self._make_observation(error_message="Episode already finished.")

        self._step_count += 1
        self._task_steps += 1

        score = self._current_task.grader(action.fix)

        if score >= 1.0:
            return self._advance_or_finish(score, error_message=None)

        if self._task_steps >= MAX_STEPS_PER_TASK:
            return self._advance_or_finish(
                score,
                error_message=(
                    f"Max attempts reached for {self._current_task.task_id}. "
                    f"Moving on. Best score: {score:.2f}."
                ),
            )

        self._show_hint = True
        return self._make_observation(
            error_message=f"Incorrect fix. Score: {score:.2f}. Try again.",
            score=score,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _advance_or_finish(self, score: float, error_message) -> MLPipelineObservation:
        self._task_index += 1
        self._task_steps = 0
        self._show_hint = False

        if self._task_index >= len(TASK_ORDER):
            self._done = True
            return MLPipelineObservation(
                task_id="complete",
                task_description="All tasks complete!",
                broken_code="",
                error_message=error_message,
                score=score,
                done=True,
                reward=float(score),
                step_count=self._step_count,
            )

        self._current_task = TASKS[TASK_ORDER[self._task_index]]
        return self._make_observation(error_message=error_message, score=score)

    def _make_observation(
        self,
        error_message=None,
        score: float = 0.0,
    ) -> MLPipelineObservation:
        return MLPipelineObservation(
            task_id=self._current_task.task_id,
            task_description=self._current_task.description,
            broken_code=self._current_task.broken_code,
            error_message=error_message,
            hint=self._current_task.hint if self._show_hint else None,
            score=score,
            done=self._done,
            reward=float(score),
            step_count=self._step_count,
        )