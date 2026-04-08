"""
ML Pipeline Debugger — FastAPI Server
create_fastapi_app expects a factory callable, not an instance.
"""

from openenv.core.env_server import create_fastapi_app
from ml_pipeline_env.models import MLPipelineAction, MLPipelineObservation
from server.ml_pipeline_environment import MLPipelineEnvironment

# This factory function creates the WebSocket endpoints and session 
# management required by the OpenEnv validator.
env=MLPipelineEnvironment()
app = create_fastapi_app(
    env,
    MLPipelineAction,
    MLPipelineObservation,
)

@app.get("/")
def health_check():
    return {"status": "ok", "info": "ML Pipeline Debugger is Running"}