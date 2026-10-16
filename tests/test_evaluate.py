"""Tests for model evaluation."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from purchase_propensity.models.evaluate import compute_metrics, evaluate
from purchase_propensity.models.train import (
    TARGET_COLUMN,
    split_features_and_target,
    train_model,
)

METRIC_NAMES = {"accuracy", "precision", "recall", "f1", "roc_auc"}


@pytest.fixture
def prepared_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            TARGET_COLUMN: [0, 0, 0, 0, 1, 1, 1, 1],
        },
    )


def test_compute_metrics_returns_expected_metrics(
    prepared_dataframe: pd.DataFrame,
) -> None:
    features, target = split_features_and_target(prepared_dataframe)
    model = train_model(features, target, {"n_estimators": 10})

    metrics = compute_metrics(model, features, target)

    assert set(metrics) == METRIC_NAMES
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_evaluate_writes_metrics_file(
    prepared_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    features, target = split_features_and_target(prepared_dataframe)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "metrics" / "metrics.json"
    joblib.dump(train_model(features, target, {"n_estimators": 10}), model_path)
    prepared_dataframe.to_csv(test_path, index=False)

    evaluate(model_path, test_path, output_path)

    assert set(json.loads(output_path.read_text())) == METRIC_NAMES


def test_evaluate_requires_existing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        evaluate(tmp_path / "missing.joblib", tmp_path / "t.csv", tmp_path / "m.json")
