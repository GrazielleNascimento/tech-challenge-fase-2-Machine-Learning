"""Tests for the MLflow Model Registry promotion."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
from sklearn.dummy import DummyClassifier

from purchase_propensity.models.registry import (
    CHAMPION_ALIAS,
    SELECTION_METRIC,
    register,
    should_promote,
)
from purchase_propensity.models.tracking import configure_mlflow


def log_fake_run(score: float) -> str:
    configure_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_metric(SELECTION_METRIC, score)
        model = DummyClassifier().fit([[0.0], [1.0]], [0, 1])
        mlflow.sklearn.log_model(model, name="model")
    return run.info.run_id


def run_register(run_id: str, tmp_path: Path) -> dict[str, object]:
    run_info_path = tmp_path / "run.json"
    run_info_path.write_text(json.dumps({"run_id": run_id}))
    return register(run_info_path, tmp_path / "registry.json")


def test_should_promote_without_champion() -> None:
    assert should_promote(0.5, None)


def test_should_promote_better_candidate() -> None:
    assert should_promote(0.8, 0.7)


def test_should_not_promote_worse_candidate() -> None:
    assert not should_promote(0.6, 0.7)


@pytest.mark.usefixtures("mlflow_tracking")
def test_register_promotes_first_run(tmp_path: Path) -> None:
    run_id = log_fake_run(0.7)

    summary = run_register(run_id, tmp_path)

    assert summary["promoted"]
    assert summary["champion_run_id"] == run_id
    assert str(summary["champion_version"]) == "1"


@pytest.mark.usefixtures("mlflow_tracking")
def test_register_keeps_champion_when_candidate_is_worse(tmp_path: Path) -> None:
    champion_run_id = log_fake_run(0.8)
    run_register(champion_run_id, tmp_path)

    summary = run_register(log_fake_run(0.6), tmp_path)

    assert not summary["promoted"]
    assert summary["champion_run_id"] == champion_run_id
    assert CHAMPION_ALIAS == "champion"


def test_should_not_promote_tie() -> None:
    assert not should_promote(0.7, 0.7)
