"""Shared pytest fixtures."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Return a minimal valid synthetic dataset."""
    return pd.DataFrame(
        {
            "Administrative": [1, 2, 3],
            "Administrative_Duration": [10.0, 20.0, 30.0],
            "Informational": [0, 1, 2],
            "Informational_Duration": [0.0, 5.0, 10.0],
            "ProductRelated": [3, 4, 5],
            "ProductRelated_Duration": [30.0, 40.0, 50.0],
            "BounceRates": [0.1, 0.2, 0.3],
            "ExitRates": [0.2, 0.3, 0.4],
            "PageValues": [0.0, 10.0, 20.0],
            "SpecialDay": [0.0, 0.5, 1.0],
            "Month": ["May", "Jun", "Jul"],
            "OperatingSystems": [1, 2, 3],
            "Browser": [1, 2, 3],
            "Region": [1, 2, 3],
            "TrafficType": [1, 2, 3],
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
                "Other",
            ],
            "Weekend": [True, False, True],
            "Revenue": [False, True, False],
        },
    )


@pytest.fixture
def mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate MLflow in a temporary SQLite store and artifact directory."""
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    return tmp_path
