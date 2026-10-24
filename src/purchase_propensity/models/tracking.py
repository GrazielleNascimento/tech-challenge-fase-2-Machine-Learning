"""Log a training run (parameters, metrics and model) to MLflow Tracking."""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.models.train import split_features_and_target
from purchase_propensity.utils.config import CONFIG
from purchase_propensity.utils.logger import get_logger

MLFLOW_CONFIG = CONFIG["mlflow"]
MODEL_ARTIFACT_NAME = "model"
INPUT_EXAMPLE_ROWS = 5
# skops (default MLflow serializer) only loads allow-listed types.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

logger = get_logger(__name__)


def configure_mlflow() -> None:
    """Point MLflow to the tracking server and select the project experiment.

    ``MLFLOW_TRACKING_URI`` (environment or ``.env``) takes precedence over
    ``mlflow.tracking_uri`` from the configuration file.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    logger.info("MLflow tracking URI: %s", tracking_uri)


def build_run_params(model: BaseEstimator) -> dict[str, object]:
    """Collect the parameters that describe a training run.

    Args:
        model: Fitted estimator whose hyperparameters are logged.

    Returns:
        Model hyperparameters plus the global settings that affect training.
    """
    return {
        **model.get_params(),
        "model_class": type(model).__name__,
        "random_seed": CONFIG["project"]["random_seed"],
        "test_size": CONFIG["model"]["test_size"],
    }


def log_run(
    model: BaseEstimator,
    metrics: dict[str, float],
    test_data: pd.DataFrame,
) -> str:
    """Log parameters, metrics, evaluation dataset and model in a new run.

    Args:
        model: Fitted estimator.
        metrics: Evaluation metrics computed on the test set.
        test_data: Prepared test set used for evaluation.

    Returns:
        The identifier of the MLflow run.
    """
    features, _ = split_features_and_target(test_data)
    with mlflow.start_run() as run:
        mlflow.log_params(build_run_params(model))
        mlflow.log_metrics(metrics)
        dataset = mlflow.data.from_pandas(test_data, name="test")
        mlflow.log_input(dataset, context="evaluation")
        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            input_example=features.head(INPUT_EXAMPLE_ROWS).astype("float64"),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    logger.info("Run registrada no MLflow: %s", run.info.run_id)
    return run.info.run_id


def save_run_info(run_id: str, output_path: Path) -> None:
    """Write the run identifier so later pipeline stages can find it.

    Args:
        run_id: MLflow run identifier.
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_info = {"run_id": run_id, "experiment": MLFLOW_CONFIG["experiment_name"]}
    output_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")


def track(
    model_path: Path,
    metrics_path: Path,
    test_path: Path,
    output_path: Path,
) -> str:
    """Log the artifacts produced by ``train`` and ``evaluate`` to MLflow.

    Args:
        model_path: Path to the serialized model.
        metrics_path: Path to the evaluation metrics JSON.
        test_path: Path to the prepared test CSV.
        output_path: Destination JSON file for the run information.

    Returns:
        The identifier of the MLflow run.
    """
    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    configure_mlflow()
    run_id = log_run(model, metrics, load_dataset(test_path))
    save_run_info(run_id, output_path)
    return run_id


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Log the run to MLflow.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run experiment tracking."""
    args = parse_arguments()
    track(
        model_path=args.model,
        metrics_path=args.metrics,
        test_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
