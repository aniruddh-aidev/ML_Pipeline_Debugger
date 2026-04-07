"""
ML Pipeline Debugger — FastAPI Server
create_fastapi_app expects a factory callable, not an instance.
"""

from openenv.core.env_server import create_fastapi_app
from ml_pipeline_env.models import MLPipelineAction, MLPipelineObservation
from server.ml_pipeline_environment import MLPipelineEnvironment

app = create_fastapi_app(
    MLPipelineEnvironment,   # factory callable — openenv instantiates per session
    MLPipelineAction,
    MLPipelineObservation,
)
