"""
ML Pipeline Debugger — FastAPI Server
"""

from openenv.core.env_server import create_fastapi_app
from ml_pipeline_env.models import MLPipelineAction, MLPipelineObservation
from server.ml_pipeline_environment import MLPipelineEnvironment

app = create_fastapi_app(
    MLPipelineEnvironment,
    MLPipelineAction,
    MLPipelineObservation,
)

@app.get("/")
def health_check():
    return {"status": "ok", "info": "ML Pipeline Debugger is Running"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()