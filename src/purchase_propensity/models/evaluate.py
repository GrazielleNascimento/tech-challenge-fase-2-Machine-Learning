"""Evaluate the trained classifier on the test set."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.models.train import split_features_and_target
from purchase_propensity.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    model: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics for a fitted model.

    Args:
        model: Fitted classifier exposing ``predict`` and ``predict_proba``.
        features: Evaluation features.
        target: True labels.

    Returns:
        Accuracy, precision, recall, F1 and ROC-AUC of the positive class.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Write metrics to a JSON file.

    Args:
        metrics: Metric names and values.
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metricas salvas em %s.", output_path)


def evaluate(model_path: Path, test_path: Path, output_path: Path) -> None:
    """Evaluate the saved model on the prepared test set.

    Args:
        model_path: Path to the serialized model.
        test_path: Path to the prepared test CSV.
        output_path: Destination JSON file for the metrics.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    model = joblib.load(model_path)
    test_data = load_dataset(test_path)
    logger.info("Dados de teste carregados: %d linhas.", len(test_data))

    features, target = split_features_and_target(test_data)
    metrics = compute_metrics(model, features, target)
    logger.info("Metricas no conjunto de teste: %s", metrics)

    save_metrics(metrics, output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the classifier.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run model evaluation."""
    args = parse_arguments()
    evaluate(
        model_path=args.model,
        test_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
