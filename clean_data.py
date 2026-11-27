"""Reproduce the notebook's cleaning and preprocessing export from the supplied CSV."""

from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
SOURCE_PATH = DATA_DIR / "Employee_Salaries_raw.csv"
HEADER = "Department,Department_Name,Division,Gender,Base_Salary,Overtime_Pay,Longevity_Pay,Grade"
RAW_COLUMNS = [
    "Department", "Department_Name", "Division", "Gender", "Base_Salary",
    "Overtime_Pay", "Longevity_Pay", "Grade",
]
CATEGORICAL_COLUMNS = ["Department", "Department_Name", "Division", "Gender", "Grade"]


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Also support the originally supplied workspace filename.
        path = PROJECT_DIR.parent / "Employee_Salaries.csv"
    text = path.read_text(encoding="utf-8-sig")
    header_position = text.find(HEADER)
    if header_position < 0:
        raise ValueError(f"Could not find the expected dataset header in {path}")
    frame = pd.read_csv(
        StringIO(text[header_position:]),
        dtype={column: "object" for column in CATEGORICAL_COLUMNS},
    )
    if frame.columns.tolist() != RAW_COLUMNS:
        raise ValueError(f"Unexpected CSV columns: {frame.columns.tolist()}")
    for column in ["Base_Salary", "Overtime_Pay", "Longevity_Pay"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def notebook_split(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Match train_test_split(test_size=.2, random_state=42) row ordering."""
    test_count = math.ceil(length * 0.2)
    permutation = np.random.RandomState(42).permutation(length)
    test_indices = permutation[:test_count]
    train_indices = permutation[test_count:]
    return train_indices, test_indices


def clean(frame: pd.DataFrame):
    features = frame.drop(columns="Base_Salary")
    target = frame["Base_Salary"]
    train_indices, test_indices = notebook_split(len(frame))
    x_train, x_test = features.iloc[train_indices].copy(), features.iloc[test_indices].copy()
    y_train, y_test = target.iloc[train_indices].copy(), target.iloc[test_indices].copy()
    numeric = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = x_train.select_dtypes(include=["object"]).columns.tolist()
    fill_values: dict[str, float | str] = {}

    for column in numeric:
        fill = float(x_train[column].median())
        fill_values[column] = fill
        x_train[column] = x_train[column].fillna(fill)
        x_test[column] = x_test[column].fillna(fill)
    for column in categorical:
        modes = x_train[column].mode(dropna=True)
        fill = str(modes.iloc[0]) if len(modes) else "Unknown"
        fill_values[column] = fill
        x_train[column] = x_train[column].fillna(fill)
        x_test[column] = x_test[column].fillna(fill)

    # Match the notebook: remove exact duplicates from the training frame only.
    train_frame = x_train.copy()
    train_frame["Base_Salary"] = y_train
    train_frame = train_frame.drop_duplicates()
    x_train = train_frame.drop(columns="Base_Salary")
    y_train = train_frame["Base_Salary"]

    clip_bounds: dict[str, list[float]] = {}
    for column in ["Overtime_Pay", "Longevity_Pay"]:
        q1, q3 = x_train[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        clip_bounds[column] = [lower, upper]
        x_train[column] = x_train[column].clip(lower, upper)
        x_test[column] = x_test[column].clip(lower, upper)

    # These encoded matrices reproduce the notebook's one-hot and left alignment.
    x_train_encoded = pd.get_dummies(x_train, columns=categorical, dtype=int)
    x_test_encoded = pd.get_dummies(x_test, columns=categorical, dtype=int)
    x_train_encoded, x_test_encoded = x_train_encoded.align(
        x_test_encoded, join="left", axis=1, fill_value=0
    )
    matrix = x_train_encoded.to_numpy(dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0

    cleaned = pd.concat(
        [x_train.assign(Base_Salary=y_train), x_test.assign(Base_Salary=y_test)],
        axis=0,
        ignore_index=True,
    )[RAW_COLUMNS]
    preprocessing = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "feature_columns": x_train_encoded.columns.tolist(),
        "clip_bounds": clip_bounds,
        "categorical_features": categorical,
        "fill_values": fill_values,
        "target": "Base_Salary",
        "train_test_split": {"test_size": 0.2, "random_state": 42},
    }
    return cleaned, preprocessing


def main() -> None:
    raw = load_raw_csv(SOURCE_PATH)
    cleaned, preprocessing = clean(raw)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = DATA_DIR / "cleaned_employee_salary.csv"
    output_json = MODEL_DIR / "inference_preprocessing.json"
    cleaned.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(preprocessing), encoding="utf-8")
    print(f"Raw rows: {len(raw):,}")
    print(f"Training duplicates removed: {len(raw) - len(cleaned):,} total rows removed from split")
    print(f"Cleaned rows: {len(cleaned):,}; missing cells: {int(cleaned.isna().sum().sum())}")
    print(f"Saved: {output_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
