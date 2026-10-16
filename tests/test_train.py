"""Tests for model training."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from purchase_propensity.models.train import (
    TARGET_COLUMN,
    build_model,
    split_features_and_target,
    train,
    train_model,
)


@pytest.fixture
def prepared_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            TARGET_COLUMN: [0, 0, 0, 0, 1, 1, 1, 1],
        },
    )


def test_split_features_and_target(prepared_dataframe: pd.DataFrame) -> None:
    features, target = split_features_and_target(prepared_dataframe)

    assert TARGET_COLUMN not in features.columns
    assert target.name == TARGET_COLUMN


def test_split_features_and_target_requires_target() -> None:
    with pytest.raises(ValueError, match="Target column not found"):
        split_features_and_target(pd.DataFrame({"feature_a": [1, 2]}))


def test_build_model_uses_fixed_seed_and_custom_params() -> None:
    model = build_model({"n_estimators": 5})

    assert model.random_state == 42  # noqa: PLR2004
    assert model.n_estimators == 5  # noqa: PLR2004


def test_train_model_is_reproducible(prepared_dataframe: pd.DataFrame) -> None:
    features, target = split_features_and_target(prepared_dataframe)
    params = {"n_estimators": 10}

    first = train_model(features, target, params).predict_proba(features)
    second = train_model(features, target, params).predict_proba(features)

    assert (first == second).all()


def test_train_saves_loadable_model(
    prepared_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "models" / "model.joblib"
    prepared_dataframe.to_csv(input_path, index=False)

    train(input_path, output_path)

    assert isinstance(joblib.load(output_path), RandomForestClassifier)
