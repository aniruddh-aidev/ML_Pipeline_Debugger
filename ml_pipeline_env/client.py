"""
ML Pipeline Debugger — OpenEnv Client
Uses WebSocket-based EnvClient (the correct base in openenv 0.2.x).
Server response format: {"observation": {...}, "reward": float, "done": bool}
"""

from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import MLPipelineAction, MLPipelineObservation


class MLPipelineEnv(EnvClient[MLPipelineAction, MLPipelineObservation, State]):
    """Client for the ML Pipeline Debugger environment."""

    def _step_payload(self, action: MLPipelineAction) -> Dict[str, Any]:
        return {
            "fix": action.fix,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MLPipelineObservation]:
        # Server wraps obs in {"observation": {...}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", payload)
        obs = MLPipelineObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.score),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(**payload)
