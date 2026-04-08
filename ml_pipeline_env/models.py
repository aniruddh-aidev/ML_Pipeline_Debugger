"""
ML Pipeline Debugger — OpenEnv Models
Action and Observation definitions.
'done' and 'reward' are already on Observation base — do NOT redefine them.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class MLPipelineAction(Action):
    """Action taken by the agent to fix the broken ML pipeline."""

    fix: str = Field(
        ...,
        description=(
            "The corrected Python code or a specific fix instruction. "
            "Can be a full corrected script or a targeted patch."
        ),
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Optional: agent's reasoning for the fix.",
    )


class MLPipelineObservation(Observation):
    """
    What the agent sees after each step.
    Note: 'done' and 'reward' are inherited from Observation base class.
    """

    task_id: str = Field(..., description="Which task is currently active.")
    task_description: str = Field(..., description="Plain-English description of the task.")
    broken_code: str = Field(..., description="The broken ML pipeline code to fix.")
    error_message: Optional[str] = Field(
        default=None,
        description="Runtime error or validation message from the last action.",
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint shown after a failed attempt.",
    )
    score: float = Field(
        default=0.001,
        description="Current score for this task (0.0–1.0).",
    )
    step_count: int = Field(default=0, description="Steps taken so far in this episode.")
