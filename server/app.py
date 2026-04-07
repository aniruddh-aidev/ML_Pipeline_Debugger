"""
ML Pipeline Debugger — FastAPI Server
create_fastapi_app expects a factory callable, not an instance.
"""

from ml_pipeline_env.models import MLPipelineAction, MLPipelineObservation
from server.ml_pipeline_environment import MLPipelineEnvironment, create_fastapi_app

# This line is the most important. 
# Uvicorn looks for the variable named 'app'
app = create_fastapi_app(
    MLPipelineEnvironment,
    MLPipelineAction,
    MLPipelineObservation
)
