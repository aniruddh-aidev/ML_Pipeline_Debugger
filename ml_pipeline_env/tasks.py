"""
ML Pipeline Debugger — Task Bank (Optimized)
Three real-world ML pipeline bugs: easy → medium → hard.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict


def _clamp(score: float) -> float:
    """Scores must be strictly in (0.0, 1.0)."""
    score = round(min(max(score, 0.0), 1.0), 3)
    if score <= 0.0: return 0.001
    if score >= 1.0: return 0.999
    return score


@dataclass
class Task:
    task_id: str
    description: str
    broken_code: str
    fixed_code: str
    hint: str
    grader: Callable[[str], float] = field(repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY: Data Leakage
# ─────────────────────────────────────────────────────────────────────────────
TASK_EASY_CODE = textwrap.dedent("""\
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv("data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # BUG: scaler is fit on ALL data before splitting — data leakage!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
""")

TASK_EASY_FIXED = textwrap.dedent("""\
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv("data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
""")


def grade_easy(agent_fix: str) -> float:
    fix = agent_fix.lower()

    # Leakage still present: fit_transform called on full X before split
    leakage = bool(
        re.search(r"fit_transform\s*\(\s*(x=)?x\s*\)", fix)
        or re.search(r"\.fit\s*\(\s*(x=)?x\s*\)", fix)
    )
    if leakage:
        return 0.001

    has_split = "train_test_split" in fix

    # Check split happens before any fitting
    split_idx = fix.find("train_test_split")
    fit_idx   = fix.find("fit")
    has_split_first = has_split and (split_idx < fit_idx if fit_idx != -1 else True)

    # Test set uses .transform() only (not fit_transform)
    has_transform_test = bool(
        re.search(r"x_test\s*=\s*\w+\.transform\s*\(", fix)
        or re.search(r"\.transform\s*\(\s*(x=)?x_test", fix)
        or ("transform" in fix and "fit_transform" not in fix.split("x_test")[1] if "x_test" in fix else False)
    )

    # Train uses fit_transform
    has_fit_train = bool(
        re.search(r"x_train\s*=\s*\w+\.fit_transform\s*\(", fix)
        or re.search(r"fit_transform\s*\(\s*(x=)?x_train", fix)
    )

    # Also accept: scaler.fit(X_train) then scaler.transform(X_train)
    has_fit_then_transform = bool(
        re.search(r"\.fit\s*\(\s*(x=)?x_train", fix)
        and re.search(r"\.transform\s*\(\s*(x=)?x_train", fix)
    )

    score = 0.0
    if has_split_first:       score += 0.4
    if has_fit_train or has_fit_then_transform: score += 0.3
    if has_transform_test:    score += 0.3

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Silent Encoding Error
# ─────────────────────────────────────────────────────────────────────────────
TASK_MEDIUM_CODE = textwrap.dedent("""\
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    df = pd.read_csv("customers.csv")

    # BUG: 'churn' column is read as string ('True'/'False').
    # Assigning directly to int without explicit conversion silently maps
    # all values to NaN, then fills with 0 — all labels become 0.
    df["churn"] = df["churn"].astype(int)

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
""")

TASK_MEDIUM_FIXED = textwrap.dedent("""\
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    df = pd.read_csv("customers.csv")

    df["churn"] = df["churn"].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
""")


def grade_medium(agent_fix: str) -> float:
    fix = agent_fix.lower()

    # Still broken: direct astype(int) with no mapping
    still_broken = bool(
        re.search(r"churn.*?\.astype\s*\(\s*['\"]?int['\"]?\s*\)", fix)
        and not any(x in fix for x in [".map", ".replace", ".apply", "astype(bool)", "lambda"])
    )
    if still_broken:
        return 0.001

    uses_map     = bool(re.search(r"\.map\s*\(", fix) and ("true" in fix or "false" in fix))
    uses_bool    = bool(re.search(r"astype\s*\(\s*['\"]?bool['\"]?\s*\)", fix))
    uses_replace = bool(re.search(r"\.replace\s*\(", fix) and ("true" in fix or "false" in fix))
    uses_lambda  = bool("lambda" in fix and ("true" in fix or "1" in fix))
    uses_apply   = bool(".apply" in fix and ("true" in fix or "1" in fix))
    uses_where   = bool("np.where" in fix and "true" in fix)

    if uses_map:                          return 0.999
    if uses_bool:                         return 0.850
    if uses_replace or uses_where:        return 0.700
    if uses_lambda or uses_apply:         return 0.600

    # Partial: mentions the problem
    if "churn" in fix and any(k in fix for k in ["true", "false", "bool", "map", "replace"]):
        return 0.300

    return 0.001


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD: PyTorch Shape Mismatch + Wrong Loss
# ─────────────────────────────────────────────────────────────────────────────
TASK_HARD_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(200, 16)
    y = torch.randint(0, 3, (200,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                # BUG 1: output is 1 neuron — wrong for 3-class classification
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP()
    # BUG 2: BCEWithLogitsLoss is for binary classification, not multi-class
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for xb, yb in loader:
            preds = model(xb)
            # BUG 3: shape mismatch
            loss = criterion(preds, yb.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
""")

TASK_HARD_FIXED = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(200, 16)
    y = torch.randint(0, 3, (200,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
""")


def grade_hard(agent_fix: str) -> float:
    fix = agent_fix.lower()

    # Fix 1: output layer has 3 neurons — flexible regex
    fixed_output = bool(
        re.search(r"linear\s*\(\s*(\w+\s*=\s*)?\d+\s*,\s*(\w+\s*=\s*)?3\s*\)", fix)
        or re.search(r"out_features\s*=\s*3", fix)
        or re.search(r"linear\s*\(.*?,\s*3\s*\)", fix)
    )

    # Fix 2: CrossEntropyLoss
    fixed_loss = bool(re.search(r"crossentropyloss", fix))

    # Fix 3: no unsqueeze + no float cast on yb
    fixed_call = bool(
        (fixed_loss and "unsqueeze" not in fix and ".float()" not in fix)
        or re.search(r"criterion\s*\(\s*\w+\s*,\s*yb\s*\)", fix)
        or re.search(r"loss\s*=\s*criterion\s*\(\s*preds\s*,\s*yb\s*\)", fix)
    )

    score = 0.0
    if fixed_output: score += 0.34
    if fixed_loss:   score += 0.33
    if fixed_call:   score += 0.33

    return _clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, Task] = {
    "task_easy": Task(
        task_id="task_easy",
        description=(
            "Fix the data leakage bug: the StandardScaler is being fit on the "
            "entire dataset BEFORE the train/test split. Move the split before "
            "scaling, fit the scaler ONLY on training data, and transform the "
            "test set separately."
        ),
        broken_code=TASK_EASY_CODE,
        fixed_code=TASK_EASY_FIXED,
        hint=(
            "Hint: call train_test_split() BEFORE StandardScaler.fit_transform(). "
            "Use .fit_transform(X_train) and .transform(X_test)."
        ),
        grader=grade_easy,
    ),
    "task_medium": Task(
        task_id="task_medium",
        description=(
            "Fix the silent encoding error: the 'churn' column contains string "
            "values 'True' and 'False' (not Python booleans). Calling .astype(int) "
            "directly silently produces NaN then 0 for all rows. Fix the conversion "
            "so labels are correctly 1 and 0."
        ),
        broken_code=TASK_MEDIUM_CODE,
        fixed_code=TASK_MEDIUM_FIXED,
        hint=(
            "Hint: use .map({'True': 1, 'False': 0}) before .astype(int), "
            "or .astype(bool).astype(int)."
        ),
        grader=grade_medium,
    ),
    "task_hard": Task(
        task_id="task_hard",
        description=(
            "Fix three bugs in this PyTorch multi-class classifier: "
            "(1) the output layer has 1 neuron instead of 3, "
            "(2) BCEWithLogitsLoss is used instead of CrossEntropyLoss, "
            "(3) the loss call has a shape mismatch due to unsqueeze+float cast. "
            "Fix all three."
        ),
        broken_code=TASK_HARD_CODE,
        fixed_code=TASK_HARD_FIXED,
        hint=(
            "Hint: change nn.Linear(64, 1) to nn.Linear(64, 3), "
            "use nn.CrossEntropyLoss(), and call criterion(preds, yb) directly."
        ),
        grader=grade_hard,
    ),
}

tasks = list(TASKS.values())