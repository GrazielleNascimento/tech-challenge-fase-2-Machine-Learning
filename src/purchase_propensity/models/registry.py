"""Register and promote the best model in the MLflow Model Registry.

The candidate is the run logged by the ``track`` stage. It becomes a new
model version and receives the champion alias only when it beats the current
champion on the selection metric (champion/challenger strategy).
"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from purchase_propensity.models.tracking import MLFLOW_CONFIG, configure_mlflow
from purchase_propensity.utils.logger import get_logger

MODEL_NAME = MLFLOW_CONFIG["registered_model_name"]
CHAMPION_ALIAS = MLFLOW_CONFIG["champion_alias"]
SELECTION_METRIC = MLFLOW_CONFIG["selection_metric"]
MIN_IMPROVEMENT = MLFLOW_CONFIG["min_improvement"]

logger = get_logger(__name__)


def should_promote(candidate_score: float, champion_score: float | None) -> bool:
    """Decide whether the candidate model replaces the current champion.

    Args:
        candidate_score: Selection metric of the candidate run.
        champion_score: Selection metric of the current champion, or ``None``
            when no model has been promoted yet.

    Returns:
        ``True`` when the candidate should become the new champion.
    """
    if champion_score is None:
        return True
    return candidate_score > champion_score + MIN_IMPROVEMENT


def get_run_score(client: MlflowClient, run_id: str) -> float:
    """Read the selection metric of a run.

    Args:
        client: MLflow client.
        run_id: MLflow run identifier.

    Returns:
        The value of the selection metric.

    Raises:
        KeyError: If the run did not log the selection metric.
    """
    metrics = client.get_run(run_id).data.metrics
    if SELECTION_METRIC not in metrics:
        msg = f"Run {run_id} has no metric '{SELECTION_METRIC}'."
        raise KeyError(msg)
    return metrics[SELECTION_METRIC]


def get_champion_score(client: MlflowClient) -> float | None:
    """Read the selection metric of the current champion version.

    Args:
        client: MLflow client.

    Returns:
        The champion's score, or ``None`` if there is no champion yet.
    """
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return get_run_score(client, champion.run_id)


def get_model_uri(client: MlflowClient, run_id: str) -> str:
    """Build the URI of the model logged by a run.

    Args:
        client: MLflow client.
        run_id: MLflow run identifier.

    Returns:
        The ``models:/<model_id>`` URI of the logged model.

    Raises:
        ValueError: If the run did not log a model.
    """
    model_outputs = client.get_run(run_id).outputs.model_outputs
    if not model_outputs:
        msg = f"Run {run_id} has no logged model."
        raise ValueError(msg)
    return f"models:/{model_outputs[0].model_id}"


def promote_run(client: MlflowClient, run_id: str) -> int:
    """Register the run's model as a new version and mark it as champion.

    Args:
        client: MLflow client.
        run_id: MLflow run identifier.

    Returns:
        The new model version.
    """
    model_uri = get_model_uri(client, run_id)
    version = mlflow.register_model(model_uri, MODEL_NAME).version
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(
        MODEL_NAME, version, "selection_metric", SELECTION_METRIC
    )
    logger.info("Modelo %s v%s promovido a '%s'.", MODEL_NAME, version, CHAMPION_ALIAS)
    return version


def register_best_model(run_id: str) -> dict[str, object]:
    """Promote the candidate run when it beats the current champion.

    Args:
        run_id: MLflow run identifier of the candidate.

    Returns:
        A summary of the registry decision.
    """
    client = MlflowClient()
    candidate_score = get_run_score(client, run_id)
    champion_score = get_champion_score(client)
    promoted = should_promote(candidate_score, champion_score)
    if promoted:
        promote_run(client, run_id)
    else:
        logger.info("Champion mantido: %s <= %s.", candidate_score, champion_score)
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    return {
        "candidate_run_id": run_id,
        "candidate_score": candidate_score,
        "promoted": promoted,
        "selection_metric": SELECTION_METRIC,
        "model_name": MODEL_NAME,
        "champion_version": champion.version,
        "champion_run_id": champion.run_id,
    }


def register(run_info_path: Path, output_path: Path) -> dict[str, object]:
    """Run the registry step for the run recorded by the ``track`` stage.

    Args:
        run_info_path: JSON file written by the ``track`` stage.
        output_path: Destination JSON file for the registry decision.

    Returns:
        A summary of the registry decision.
    """
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    configure_mlflow()
    summary = register_best_model(run_info["run_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Register the best model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run model registration."""
    args = parse_arguments()
    register(run_info_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
