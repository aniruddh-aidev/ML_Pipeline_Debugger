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

from fastapi import Request

@app.post("/grader")
async def grader(request: Request):
    body = await request.json()
    task_id = body.get("task_id", "")
    fix = body.get("fix", "") or body.get("action", {}).get("fix", "")
    
    from ml_pipeline_env.tasks import TASKS
    if task_id not in TASKS:
        # Return scores for all tasks if no specific task_id
        return {
            "tasks": [
                {"id": tid, "score": task.grader(fix)} 
                for tid, task in TASKS.items()
            ]
        }
    
    score = TASKS[task_id].grader(fix)
    return {"id": task_id, "score": score}

@app.get("/grader")
def list_graders():
    from ml_pipeline_env.tasks import TASKS
    return {
        "tasks": [
            {"id": tid, "score_range": [0.001, 0.999]}
            for tid in TASKS.keys()
        ]
    }

@app.get("/tasks")
def list_tasks():
    from ml_pipeline_env.tasks import TASKS
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": ["easy", "medium", "hard"][i] if i < 3 else "medium",
                "has_grader": True,
                "score_range": [0.001, 0.999]
            }
            for i, tid in enumerate(TASKS.keys())
        ]
    }

if __name__ == "__main__":
    main()

