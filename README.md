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

---

## 🔗 Project Links
* **Live Environment (Hugging Face):** [https://huggingface.co/spaces/annir241/ML_Pipeline_Debugger](https://huggingface.co/spaces/annir241/ML_Pipeline_Debugger)
* **Source Code (GitHub):** [https://github.com/aniruddh-aidev/ML_Pipeline_Debugger](https://github.com/aniruddh-aidev/ML_Pipeline_Debugger)

---

## 🛠️ Tech Stack
* **Framework:** [OpenEnv](https://github.com/openenv-core) (Standardized RL-style environment factory)
* **Backend:** FastAPI, Uvicorn (Asynchronous API & WebSockets)
* **Deep Learning:** PyTorch (Task-specific models and graders)
* **Deployment:** Docker, Hugging Face Spaces
* **Languages:** Python 3.10+

---

## Project Structure
```
ml_pipeline_debugger/
├── inference.py                # Hackathon submission script
├── test_local.py               # Local smoke test
├── validate-submission.sh      # Submission validation script
├── Dockerfile                  # Container configuration
├── README.md                   # This documentation
├── ml_pipeline_env/            # Environment Package
│   ├── tests/                  # Unit and Integration tests
│   │   ├── test_environment.py
│   │   ├── test_graders.py
│   │   └── __init__.py
│   ├── client.py               # API Client logic
│   ├── models.py               # Action/Observation schemas
│   ├── openenv.yaml            # Manifest
│   ├── pyproject.toml          # Build system config
│   ├── tasks.py                # Task definitions & Graders
│   └── __init__.py
└── server/                     # Deployment Layer
    ├── app.py                  # FastAPI Entry Point
    ├── ml_pipeline_environment.py # Environment Logic
    ├── requirements.txt        # Backend dependencies
    └── __init__.py
```

## Quick Start
```bash
# 1. Install
pip install openenv-core
pip install -e ml_pipeline_env/

# 2. Smoke test
python test_local.py

# 3. Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 4. Deploy to HF Spaces
openenv push --repo-id aniruddh-aidev/ml-pipeline-env

# 5. Run inference
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token
export HF_SPACE_URL=https://annir241-ml-pipeline-debugger.hf.space
python inference.py
```

## Tasks
| Task | Difficulty | Bug |
|------|-----------|-----|
| task_easy | Easy | Data leakage — scaler fit before train/test split |
| task_medium | Medium | Silent encoding — string 'True'/'False' cast directly to int |
| task_hard | Hard | 3 PyTorch bugs — wrong output size, wrong loss, shape mismatch |
