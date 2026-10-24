"""Load project configuration from a YAML file and the optional ``.env``."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path.cwd()
load_dotenv(PROJECT_ROOT / ".env")

config_path = PROJECT_ROOT / "configs" / "config.yaml"
with config_path.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
