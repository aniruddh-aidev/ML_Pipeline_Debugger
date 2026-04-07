---
title: Ml Pipeline Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---


# ML Pipeline Debugger — OpenEnv Environment
**Scaler × Meta PyTorch Hackathon | Round 1**

An OpenEnv environment where an LLM agent debugs broken ML pipeline scripts.
Three real-world tasks: data leakage, silent encoding errors, PyTorch shape mismatches.

## Project Structure
```
ml_pipeline_debugger/
├── inference.py                   ← Hackathon submission script (root level, mandatory)
├── test_local.py                  ← Zero-dependency smoke test (run before deploying)
├── README.md                      ← This file
└── ml_pipeline_env/               ← OpenEnv environment package
    ├── __init__.py
    ├── models.py                  ← Action + Observation data models
    ├── tasks.py                   ← 3 tasks with graders (easy/medium/hard)
    ├── client.py                  ← HTTP client for inference script
    ├── openenv.yaml               ← OpenEnv manifest
    ├── pyproject.toml             ← Package config + dependencies
    ├── README.md                  ← HF Space public docs
    ├── server/
    │   ├── app.py                 ← FastAPI server entry point
    │   ├── ml_pipeline_environment.py  ← Core reset/step/state logic
    │   ├── Dockerfile             ← Docker image definition
    │   └── requirements.txt      ← Server dependencies
    └── tests/
        ├── test_graders.py        ← Unit tests for all graders
        └── test_environment.py   ← Integration tests for episode loop
```

## Quick Start
```bash
# 1. Install
pip install openenv-core
pip install -e ml_pipeline_env/

# 2. Smoke test
python test_local.py

# 3. Run server locally
uvicorn ml_pipeline_env.server.app:app --host 0.0.0.0 --port 8000

# 4. Deploy to HF Spaces
openenv push --repo-id aniruddh-aidev/ml-pipeline-env

# 5. Run inference
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token
export HF_SPACE_URL=https://aniruddh-aidev-ml-pipeline-env.hf.space
python inference.py
```

## Tasks
| Task | Difficulty | Bug |
|------|-----------|-----|
| task_easy | Easy | Data leakage — scaler fit before train/test split |
| task_medium | Medium | Silent encoding — string 'True'/'False' cast directly to int |
| task_hard | Hard | 3 PyTorch bugs — wrong output size, wrong loss, shape mismatch |
