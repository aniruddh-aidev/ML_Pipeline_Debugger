"""
ML Pipeline Debugger — Inference Script
========================================
MANDATORY env vars:
    API_BASE_URL        LLM API endpoint
    MODEL_NAME          Model identifier
    HF_TOKEN            Hugging Face / API key
    HF_SPACE_URL        Deployed HF Space URL  (used with .sync() WebSocket client)
    LOCAL_IMAGE_NAME    Local Docker image name (used with from_docker_image(), optional)

STDOUT FORMAT (strictly required by hackathon judges):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line always emitted (even on exception), after env.close().
    - reward / score formatted to 2 decimal places in [STEP], 3 in [END].
    - done and success are lowercase: true or false.
    - error is the raw error string, or null if none.
    - No newlines within any log line.
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
HF_SPACE_URL     = os.getenv("HF_SPACE_URL", "https://aniruddh-aidev-ml-pipeline-env.hf.space")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional: use docker image instead of HF Space

# ── Config ────────────────────────────────────────────────────────────────────
TASK_NAME             = "ml_pipeline_debug"
BENCHMARK             = "ml_pipeline_env"
MAX_STEPS_PER_TASK    = 5          # must match environment MAX_STEPS_PER_TASK
TEMPERATURE           = 0.2
MAX_TOKENS            = 1024
SUCCESS_SCORE_THRESHOLD = 0.5     # episode counts as success if avg score >= this

# ── OpenAI client (required by hackathon) ────────────────────────────────────
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


# ── Structured stdout loggers (mandatory format) ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Collapse any newlines in action to keep line single-line
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    error_val   = error.replace("\n", " ") if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
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
    """Ask the LLM to fix the broken pipeline. Returns cleaned code string."""
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
        raw = "# fallback: no fix generated"

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        end   = -1 if lines[-1].strip() == "```" else len(lines)
        raw   = "\n".join(lines[1:end])

    return raw.strip() or "# empty response"


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode() -> tuple[bool, int, float, list[float]]:
    """
    Run one full episode through all 3 tasks.
    Returns: (success, total_steps, avg_score, all_rewards)
    """
    all_rewards: list[float]    = []
    task_scores: dict[str, float] = {}
    total_steps  = 0
    attempt      = 0
    current_task = ""
    obs          = None

    # Connect via WebSocket (.sync() wrapper) or docker image
    if LOCAL_IMAGE_NAME:
        env = MLPipelineEnv.from_docker_image(LOCAL_IMAGE_NAME).sync()
    else:
        env = MLPipelineEnv(base_url=HF_SPACE_URL).sync()

    try:
        result      = env.reset()
        obs         = result.observation
        current_task = obs.task_id

        for step in range(1, MAX_STEPS_PER_TASK * 3 + 1):
            # 1. Check if the PREVIOUS step finished the episode
            if obs is not None and getattr(obs, 'done', False):
                print(f"[DEBUG] Loop exit at top: obs.done is True")
                break

            total_steps = step
            if obs.task_id != current_task:
                current_task = obs.task_id
                attempt = 0
            attempt += 1

            fix = get_fix(obs.task_description, obs.broken_code, obs.hint, attempt)

            # 2. Perform the action
            result = env.step(MLPipelineAction(fix=fix))
            
            # 3. Extract status IMMEDIATELY from the fresh result
            obs = result.observation
            step_reward = float(result.reward or getattr(obs, 'score', 0.0))
            step_done = bool(getattr(obs, 'done', False))
            step_error = getattr(obs, 'error_message', None)

            all_rewards.append(step_reward)

            # Track scores
            if current_task not in task_scores or step_reward > task_scores[current_task]:
                task_scores[current_task] = step_reward

            log_step(step=step, action=fix, reward=step_reward, done=step_done, error=step_error)

            # 4. EXTREME CHECK: If either the result or the observation says we are done, STOP.
            if step_done or getattr(result, 'done', False):
                print(f"[DEBUG] Breaking loop: environment is finished.")
                break

    finally:
        env.close()

    avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    success   = avg_score >= SUCCESS_SCORE_THRESHOLD

    return success, total_steps, avg_score, all_rewards


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    success    = False
    total_steps = 0
    avg_score  = 0.0
    all_rewards: list[float] = []

    try:
        success, total_steps, avg_score, all_rewards = run_episode()
    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
    finally:
        # [END] is ALWAYS emitted, even on exception
        log_end(
            success=success,
            steps=total_steps,
            score=avg_score,
            rewards=all_rewards,
        )


if __name__ == "__main__":
    main()
