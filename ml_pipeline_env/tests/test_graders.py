"""
Unit tests for ML Pipeline Debugger graders.
Run: pytest ml_pipeline_env/tests/test_graders.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml_pipeline_env.tasks import grade_easy, grade_medium, grade_hard, TASKS
from server.ml_pipeline_environment import TASK_ORDER


class TestGradeEasy:

    def test_perfect_fix(self):
        fix = """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """
        assert grade_easy(fix) == 1.0

    def test_still_leaking(self):
        fix = """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
        """
        assert grade_easy(fix) == 0.0

    def test_partial_split_first_only(self):
        fix = """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        """
        score = grade_easy(fix)
        assert 0.0 < score < 1.0

    def test_empty_fix(self):
        assert grade_easy("") == 0.0


class TestGradeMedium:

    def test_perfect_map_fix(self):
        fix = 'df["churn"] = df["churn"].map({"True": 1, "False": 0}).astype(int)'
        assert grade_medium(fix) == 1.0

    def test_bool_chain_fix(self):
        fix = 'df["churn"] = df["churn"].astype(bool).astype(int)'
        assert grade_medium(fix) == 0.8

    def test_replace_fix(self):
        fix = 'df["churn"] = df["churn"].replace({"True": 1, "False": 0})'
        assert grade_medium(fix) == 0.6

    def test_wrong_fix_still_astype_int(self):
        fix = 'df["churn"] = df["churn"].astype(int)'
        assert grade_medium(fix) == 0.0

    def test_empty_fix(self):
        assert grade_medium("") == 0.0


class TestGradeHard:

    def test_perfect_fix_all_three(self):
        fix = """
        nn.Linear(64, 3),
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, yb)
        """
        assert grade_hard(fix) == 1.0

    def test_only_output_fixed(self):
        fix = """
        nn.Linear(64, 3)
        criterion = nn.BCEWithLogitsLoss()
        """
        score = grade_hard(fix)
        assert 0.3 <= score < 0.7

    def test_only_loss_fixed(self):
        fix = """
        nn.Linear(64, 1)
        criterion = nn.CrossEntropyLoss()
        """
        score = grade_hard(fix)
        assert 0.3 <= score < 0.7

    def test_no_fix(self):
        fix = """
        nn.Linear(64, 1)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds, yb.unsqueeze(1).float())
        """
        assert grade_hard(fix) == 0.0

    def test_empty_fix(self):
        assert grade_hard("") == 0.0


class TestTaskRegistry:

    def test_all_tasks_present(self):
        assert "task_easy" in TASKS
        assert "task_medium" in TASKS
        assert "task_hard" in TASKS

    def test_task_order_has_3(self):
        assert len(TASK_ORDER) == 3

    def test_broken_code_not_empty(self):
        for task in TASKS.values():
            assert len(task.broken_code.strip()) > 0

    def test_fixed_code_scores_1(self):
        for task in TASKS.values():
            score = task.grader(task.fixed_code)
            assert score == 1.0, f"Task '{task.task_id}' fixed_code scored {score}, expected 1.0"

    def test_broken_code_scores_0(self):
        for task in TASKS.values():
            score = task.grader(task.broken_code)
            assert score == 0.0, f"Task '{task.task_id}' broken_code scored {score}, expected 0.0"
