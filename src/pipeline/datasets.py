import json
import logging
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_dataset_from_local(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a local JSON file."""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.error("Dataset file not found: %s", file_path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Error decoding JSON in %s: %s", file_path, exc)
        raise

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {file_path}")

    return data


def load_dataset_from_hub(hub_dataset_id: str) -> List[Dict[str, Any]]:
    """Load dataset from the Hugging Face Hub using the default split."""

    try:
        logger.info("Loading dataset from Hub: %s", hub_dataset_id)
        dataset = load_dataset(hub_dataset_id, split="train")
    except Exception as exc:
        logger.error("Error loading dataset from Hub %s: %s", hub_dataset_id, exc)
        raise

    if dataset:
        return dataset.to_list()

    logger.error("No data found in dataset: %s", hub_dataset_id)
    return []


__all__ = [
    "load_dataset_from_hub",
    "load_dataset_from_local",
]

