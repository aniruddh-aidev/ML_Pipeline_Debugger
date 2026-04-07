# ML Pipeline Debugger — OpenEnv Environment

An OpenEnv environment where an LLM agent is presented with broken ML pipeline
scripts and must submit corrected fixes. Three real-world tasks of increasing difficulty.

## Environment Description

The agent receives a broken Python ML script and must return a corrected version.
After each submission the environment grades the fix and returns a score (0.0–1.0)
along with optional hints. The agent has up to 5 attempts per task.

## Tasks

| Task | Difficulty | Bug |
|---|---|---|
| `task_easy` | Easy | Data leakage — scaler fit before train/test split |
| `task_medium` | Medium | Silent encoding error — string booleans cast to int |
| `task_hard` | Hard | 3 PyTorch bugs — wrong output size, wrong loss, shape mismatch |

## Action Space

```python
MLPipelineAction(
    fix: str,              # Corrected Python code or fix instruction
    explanation: str       # Optional agent reasoning
)
```

## Observation Space

```python
MLPipelineObservation(
    task_id: str,          # Current task ID
    task_description: str, # What the agent must fix
    broken_code: str,      # The buggy script
    error_message: str,    # Feedback from last attempt
    hint: str,             # Shown after first failure
    score: float,          # 0.0–1.0
    done: bool,            # Episode complete
    step_count: int        # Total steps taken
)
```

## Reward

- `0.0` — No meaningful fix detected  
- `0.3–0.9` — Partial fix (identified the bug but incomplete correction)  
- `1.0` — Correct and complete fix

## Setup

```bash
pip install openenv-core
pip install -e .
```

## Run Locally

```bash
uvicorn ml_pipeline_env.server.app:app --host 0.0.0.0 --port 8000
```

## Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token
export HF_SPACE_URL=https://aniruddh-aidev-ml-pipeline-env.hf.space

python inference.py
```

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id aniruddh-aidev/ml-pipeline-env
```
