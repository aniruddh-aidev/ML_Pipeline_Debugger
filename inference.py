"""
ML Pipeline Debugger — Inference Script (Optimized)
=====================================================
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
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_SPACE_URL     = os.getenv("HF_SPACE_URL", "https://annir241-ml-pipeline-debugger.hf.space")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARK             = "ml_pipeline_env"
MAX_STEPS_PER_TASK    = 5
TEMPERATURE           = 0.1
MAX_TOKENS            = 2048
SUCCESS_SCORE_THRESHOLD = 0.5
TASK_IDS              = ["task_easy", "task_medium", "task_hard"]

# ── Per-task focused system prompts ───────────────────────────────────────────
TASK_PROMPTS = {
    "task_easy": textwrap.dedent("""\
        You are an expert ML engineer fixing a DATA LEAKAGE bug.
        The bug: StandardScaler is being fit on the ENTIRE dataset before train/test split.
        The fix requires exactly 3 steps:
        1. Call train_test_split() FIRST on raw X and y
        2. Call scaler.fit_transform(X_train) on training data only
        3. Call scaler.transform(X_test) on test data (NOT fit_transform)
        Return ONLY the corrected Python script. No markdown, no explanation.
    """).strip(),

    "task_medium": textwrap.dedent("""\
        You are an expert ML engineer fixing a SILENT ENCODING BUG.
        The bug: a 'churn' column contains string values 'True'/'False' (not Python booleans).
        Calling .astype(int) directly silently produces NaN then 0 for all rows.
        The fix: use .map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)
        OR use .astype(bool).astype(int)
        Return ONLY the corrected Python script. No markdown, no explanation.
    """).strip(),

    "task_hard": textwrap.dedent("""\
        You are an expert ML engineer fixing THREE PyTorch bugs in a multi-class classifier.
        Bug 1: nn.Linear(64, 1) should be nn.Linear(64, 3) — need 3 output neurons for 3 classes
        Bug 2: nn.BCEWithLogitsLoss() should be nn.CrossEntropyLoss() — wrong loss for multi-class
        Bug 3: criterion(preds, yb.unsqueeze(1).float()) should be criterion(preds, yb) — shape mismatch
        Fix ALL THREE bugs. Return ONLY the corrected Python script. No markdown, no explanation.
    """).strip(),
}

DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Python and ML engineer fixing broken ML pipeline scripts.
    Return ONLY the corrected Python script. No markdown fences, no explanation.
""").strip()

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


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

def get_fix(
    task_id: str,
    task_description: str,
    broken_code: str,
    hint: Optional[str],
    attempt: int,
    last_score: float = 0.0,
) -> str:
    system_prompt = TASK_PROMPTS.get(task_id, DEFAULT_SYSTEM_PROMPT)

    # Build increasingly specific user prompt based on attempt number
    if attempt == 1:
        user_msg = textwrap.dedent(f"""\
            Fix this broken ML pipeline script:

            {task_description}

            Broken code:
            ```python
            {broken_code}
            ```

            Return ONLY the corrected Python script.
        """).strip()
    else:
        hint_section = f"\nHint: {hint}" if hint else ""
        user_msg = textwrap.dedent(f"""\
            Your previous fix scored {last_score:.3f}/1.0 — not correct yet.
            {hint_section}

            Try again. Fix this broken ML pipeline script:
            {task_description}

            Broken code:
            ```python
            {broken_code}
            ```

            Return ONLY the corrected Python script.
        """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "# fallback"

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        end   = -1 if lines[-1].strip() == "```" else len(lines)
        raw   = "\n".join(lines[1:end])

    return raw.strip() or "# empty response"


# ── Single task episode ───────────────────────────────────────────────────────

def run_episode(task_id: str) -> tuple[bool, int, float, list[float]]:
    """Run one episode for a single task_id. Returns (success, steps, best_score, rewards)."""
    all_rewards: list[float] = []
    best_score  = 0.0
    total_steps = 0
    last_score  = 0.0

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

            fix = get_fix(
                task_id=task_id,
                task_description=obs.task_description,
                broken_code=obs.broken_code,
                hint=obs.hint,
                attempt=step,
                last_score=last_score,
            )

            result      = env.step(MLPipelineAction(fix=fix))
            obs         = result.observation
            step_reward = float(result.reward or getattr(obs, 'score', 0.0))
            step_done   = bool(getattr(obs, 'done', False))
            step_error  = getattr(obs, 'error_message', None)
            last_score  = step_reward

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