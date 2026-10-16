"""Train the purchase propensity classifier."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.utils.config import CONFIG
from purchase_propensity.utils.logger import get_logger

RANDOM_SEED = CONFIG["project"]["random_seed"]
TARGET_COLUMN = CONFIG["model"]["target_column"]
MODEL_PARAMS = CONFIG["model"]["params"]

logger = get_logger(__name__)


def split_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target.

    Args:
        dataframe: Prepared dataset containing the target column.

    Returns:
        A tuple containing the feature DataFrame and the target Series.

    Raises:
        ValueError: If the target column is missing.
    """
    if TARGET_COLUMN not in dataframe.columns:
        msg = f"Target column not found: {TARGET_COLUMN}"
        raise ValueError(msg)

    return dataframe.drop(columns=[TARGET_COLUMN]), dataframe[TARGET_COLUMN]


def build_model(params: dict | None = None) -> RandomForestClassifier:
    """Build an unfitted Random Forest with a fixed random seed.

    Args:
        params: Hyperparameters. Defaults to ``model.params`` from the config.

    Returns:
        An unfitted RandomForestClassifier.
    """
    return RandomForestClassifier(
        **(MODEL_PARAMS if params is None else params),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict | None = None,
) -> RandomForestClassifier:
    """Fit the classifier on the training data.

    Args:
        features: Training features.
        target: Training target.
        params: Optional hyperparameters overriding the configuration.

    Returns:
        The fitted model.
    """
    model = build_model(params)
    logger.info("Treinando modelo com parametros: %s", model.get_params())
    return model.fit(features, target)


def save_model(model: RandomForestClassifier, output_path: Path) -> None:
    """Persist the trained model.

    Args:
        model: Fitted model.
        output_path: Destination file for the serialized model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Modelo salvo em %s.", output_path)


def train(input_path: Path, output_path: Path) -> None:
    """Train the model on the prepared training set and save it.

    Args:
        input_path: Path to the prepared training CSV.
        output_path: Destination file for the serialized model.
    """
    train_data = load_dataset(input_path)
    logger.info("Dados de treino carregados: %d linhas.", len(train_data))

    features, target = split_features_and_target(train_data)
    model = train_model(features, target)
    save_model(model, output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train the classifier.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run model training."""
    args = parse_arguments()
    train(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
