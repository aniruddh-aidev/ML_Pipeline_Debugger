"""
Local smoke test — runs WITHOUT openenv installed.
Tests graders and environment logic directly.

Run: python test_local.py
Expected: all checks print ✅
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline_env.tasks import TASKS
from server.ml_pipeline_environment import (
    MLPipelineEnvironment,
    TASK_ORDER,
    MAX_STEPS_PER_TASK,
)
from ml_pipeline_env.models import MLPipelineAction


def check(label: str, condition: bool):
    icon = "✅" if condition else "❌"
    print(f"  {icon}  {label}")
    if not condition:
        sys.exit(1)


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ── 1. Task registry ─────────────────────────────────────────────────────────
section("1. Task Registry")

check("All 3 tasks registered", len(TASKS) == 3)
check("TASK_ORDER has 3 entries", len(TASK_ORDER) == 3)

for task_id, task in TASKS.items():
    check(f"{task_id}: broken_code not empty", len(task.broken_code.strip()) > 0)
    check(f"{task_id}: fixed_code not empty",  len(task.fixed_code.strip()) > 0)
    check(f"{task_id}: hint not empty",        len(task.hint.strip()) > 0)

# ── 2. Grader correctness ─────────────────────────────────────────────────────
section("2. Grader Correctness")

for task_id, task in TASKS.items():
    broken_score = task.grader(task.broken_code)
    fixed_score  = task.grader(task.fixed_code)
    check(f"{task_id}: broken_code scores 0.0  (got {broken_score})", broken_score == 0.0)
    check(f"{task_id}: fixed_code  scores 1.0  (got {fixed_score})",  fixed_score  == 1.0)

# ── 3. Environment: reset ─────────────────────────────────────────────────────
section("3. Environment — reset()")

env = MLPipelineEnvironment()
obs = env.reset()
check("reset() task_id == task_easy",  obs.task_id == "task_easy")
check("reset() done == False",         obs.done is False)
check("reset() step_count == 0",       obs.step_count == 0)
check("reset() no hint initially",     obs.hint is None)
check("reset() broken_code not empty", len(obs.broken_code.strip()) > 0)

# ── 4. Environment: bad actions ───────────────────────────────────────────────
section("4. Environment — Bad Actions")

env.reset()
obs = env.step(MLPipelineAction(fix="# nothing"))
check("After 1 bad action: hint shown",        obs.hint is not None)
check("After 1 bad action: step_count == 1",   obs.step_count == 1)
check("After 1 bad action: still task_easy",   obs.task_id == "task_easy")
check("After 1 bad action: score == 0.0",      obs.score == 0.0)

# Exhaust remaining attempts
for _ in range(MAX_STEPS_PER_TASK - 1):
    obs = env.step(MLPipelineAction(fix="# nothing"))

check("After exhausting attempts: advances to task_medium", obs.task_id == "task_medium")

# ── 5. Environment: perfect actions ──────────────────────────────────────────
section("5. Environment — Perfect Actions (full episode)")

env.reset()
final = None
for task_id in TASK_ORDER:
    final = env.step(MLPipelineAction(fix=TASKS[task_id].fixed_code))

check("Perfect episode: done == True",        final.done is True)
check("Perfect episode: task_id == complete", final.task_id == "complete")

# ── 6. Environment: post-done step ───────────────────────────────────────────
section("6. Environment — Step After Done")

obs = env.step(MLPipelineAction(fix="# extra step"))
check("Step after done: still done", obs.done is True)

# ── 7. State ─────────────────────────────────────────────────────────────────
section("7. Environment — State")

env.reset()
env.step(MLPipelineAction(fix="# step 1"))
env.step(MLPipelineAction(fix="# step 2"))
state = env.state
check("state.step_count == 2",         state.step_count == 2)
check("state.episode_id not empty",    len(state.episode_id) > 0)

# ── 8. Double reset ───────────────────────────────────────────────────────────
section("8. Environment — Double Reset")

env.reset()
env.step(MLPipelineAction(fix="# some step"))
obs = env.reset()
check("After re-reset: task_id == task_easy", obs.task_id == "task_easy")
check("After re-reset: step_count == 0",      obs.step_count == 0)
check("After re-reset: done == False",        obs.done is False)

# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  All checks passed! Ready to deploy. 🚀")
print(f"{'='*50}\n")
