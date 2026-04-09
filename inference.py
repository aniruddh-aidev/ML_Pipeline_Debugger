"""
ML Pipeline Debugger — Inference Script
========================================
MANDATORY env vars:
    API_BASE_URL        LLM API endpoint
    MODEL_NAME          Model identifier
    HF_TOKEN            Hugging Face / API key
    HF_SPACE_URL        Deployed HF Space URL
    LOCAL_IMAGE_NAME    Local Docker image name (optional)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import textwrap
from typing import Optional

from openai import OpenAI
from ml_pipeline_env import MLPipelineAction, MLPipelineEnv

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_SPACE_URL     = os.getenv("HF_SPACE_URL", "https://annir241-ml-pipeline-debugger.hf.space")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARK             = "ml_pipeline_env"
MAX_STEPS_PER_TASK    = 5
TEMPERATURE           = 0.2
MAX_TOKENS            = 1024
SUCCESS_SCORE_THRESHOLD = 0.5
TASK_IDS              = ["task_easy", "task_medium", "task_hard"]

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Python and ML engineer specialising in debugging machine learning pipelines.
    You will be given a broken Python script with one or more bugs.
    Your job is to return ONLY the fully corrected Python script — no explanation, no markdown fences,
    no preamble. Just the corrected code, ready to run.
    Common bugs you will encounter:
    - Data leakage: scaler/encoder fit before train/test split
    - Silent encoding errors: wrong dtype conversion (e.g. string 'True'/'False' cast directly to int)
    - PyTorch shape mismatches and wrong loss functions for multi-class classification
""").strip()


# ── Structured stdout loggers ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    error_val   = error.replace("\n", " ") if error else "null"
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ───────────────────────────────────────────────────────────

def get_fix(task_description: str, broken_code: str, hint: Optional[str], attempt: int) -> str:
    hint_section = f"\nHint: {hint}" if hint else ""
    user_msg = textwrap.dedent(f"""\
        Task (attempt {attempt}): {task_description}
        {hint_section}

        Broken code:
        ```python
        {broken_code}
        ```

        Return ONLY the corrected Python script, no markdown fences, no explanation.
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        raw = "# fallback"

    if raw.startswith("```"):
        lines = raw.splitlines()
        end   = -1 if lines[-1].strip() == "```" else len(lines)
        raw   = "\n".join(lines[1:end])

    return raw.strip() or "# empty response"


# ── Single task episode ───────────────────────────────────────────────────────

def run_episode(task_id: str) -> tuple[bool, int, float, list[float]]:
    """Run one episode for a single task_id."""
    all_rewards: list[float] = []
    best_score  = 0.0
    total_steps = 0

    if LOCAL_IMAGE_NAME:
        env = MLPipelineEnv.from_docker_image(LOCAL_IMAGE_NAME).sync()
    else:
        env = MLPipelineEnv(base_url=HF_SPACE_URL).sync()

    try:
        result = env.reset(task_id=task_id)
        obs    = result.observation

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if getattr(obs, 'done', False):
                break

            total_steps = step
            fix = get_fix(obs.task_description, obs.broken_code, obs.hint, step)

            result      = env.step(MLPipelineAction(fix=fix))
            obs         = result.observation
            step_reward = float(result.reward or getattr(obs, 'score', 0.0))
            step_done   = bool(getattr(obs, 'done', False))
            step_error  = getattr(obs, 'error_message', None)

            all_rewards.append(step_reward)
            if step_reward > best_score:
                best_score = step_reward

            log_step(step=step, action=fix, reward=step_reward, done=step_done, error=step_error)

            if step_done or getattr(result, 'done', False):
                break

    finally:
        env.close()

    success = best_score >= SUCCESS_SCORE_THRESHOLD
    return success, total_steps, best_score, all_rewards


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    for task_id in TASK_IDS:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        success     = False
        total_steps = 0
        score       = 0.0
        rewards: list[float] = []

        try:
            success, total_steps, score, rewards = run_episode(task_id)
        except Exception as exc:
            print(f"[DEBUG] Episode {task_id} failed: {exc}", flush=True)
        finally:
            log_end(success=success, steps=total_steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()