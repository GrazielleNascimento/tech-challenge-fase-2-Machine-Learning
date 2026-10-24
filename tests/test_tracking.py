"""Tests for MLflow experiment tracking."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient

from purchase_propensity.models.tracking import build_run_params, track
from purchase_propensity.models.train import (
    TARGET_COLUMN,
    split_features_and_target,
    train_model,
)

METRICS = {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}


@pytest.fixture
def prepared_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            TARGET_COLUMN: [0, 0, 0, 0, 1, 1, 1, 1],
        },
    )


@pytest.fixture
def pipeline_outputs(prepared_dataframe: pd.DataFrame, tmp_path: Path) -> Path:
    features, target = split_features_and_target(prepared_dataframe)
    joblib.dump(train_model(features, target, {"n_estimators": 5}), tmp_path / "m.pkl")
    (tmp_path / "metrics.json").write_text(json.dumps(METRICS))
    prepared_dataframe.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def test_build_run_params_includes_seed_and_model_class(
    prepared_dataframe: pd.DataFrame,
) -> None:
    features, target = split_features_and_target(prepared_dataframe)

    params = build_run_params(train_model(features, target, {"n_estimators": 5}))

    assert params["random_seed"] == 42  # noqa: PLR2004
    assert params["model_class"] == "RandomForestClassifier"
    assert params["n_estimators"] == 5  # noqa: PLR2004


@pytest.mark.usefixtures("mlflow_tracking")
def test_track_logs_params_metrics_and_model(pipeline_outputs: Path) -> None:
    run_info_path = pipeline_outputs / "run.json"

    run_id = track(
        pipeline_outputs / "m.pkl",
        pipeline_outputs / "metrics.json",
        pipeline_outputs / "test.csv",
        run_info_path,
    )

    run = MlflowClient().get_run(run_id)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert run.data.metrics == METRICS
    assert run.data.params["n_estimators"] == "5"
    assert json.loads(run_info_path.read_text())["run_id"] == run_id
    assert hasattr(model, "predict_proba")
