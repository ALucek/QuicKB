from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chunking.registry import ChunkerRegistry
from hub_upload.dataset_pusher import DatasetPusher
from prompts.question_generation import QUESTION_GENERATION_PROMPT
from synth_dataset.question_generator import QuestionGenerator
from training.train import main as train_main

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def process_chunks(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Process documents into chunks and optionally upload to the Hub."""

    if not config.chunker_config:
        raise ValueError("Chunker config must be provided to process chunks")

    chunker_cfg = config.chunker_config
    chunker_class = ChunkerRegistry.get_chunker(chunker_cfg.chunker)
    chunker = chunker_class(**chunker_cfg.chunker_arguments.copy())

    logger.info("Initialized Chunker: %s", chunker_cfg.chunker)

    base_path = Path(config.path_to_knowledgebase)
    results: List[Dict[str, Any]] = []
    total_chunks = 0

    for file_path in base_path.rglob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception as exc:
            logger.error("Error reading %s: %s", file_path, exc)
            continue

        try:
            chunks = chunker.split_text(text)
        except Exception as exc:
            logger.error("Error chunking %s: %s", file_path, exc)
            continue

        source_path = str(file_path.relative_to(base_path))
        for chunk in chunks:
            results.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": source_path,
            })

        logger.info("Created %d chunks from %s", len(chunks), file_path)
        total_chunks += len(chunks)

    logger.info("Created %d chunks in total", total_chunks)

    output_path = Path(chunker_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    if (
        config.hub_username
        and chunker_cfg.upload_config
        and chunker_cfg.upload_config.push_to_hub
    ):
        _push_chunks_to_hub(config)

    return results


def _push_chunks_to_hub(config: PipelineConfig) -> None:
    chunker_cfg = config.chunker_config
    if not chunker_cfg:
        return

    if not config.hub_username:
        logger.warning("Skipping chunk upload: no Hugging Face username configured.")
        return

    token = config.hub_token or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("Skipping chunk upload: no Hugging Face token available.")
        return

    try:
        pusher = DatasetPusher(username=config.hub_username, token=token)

        repository_id = (
            chunker_cfg.upload_config.hub_dataset_id
            if chunker_cfg.upload_config and chunker_cfg.upload_config.hub_dataset_id
            else f"{config.hub_username}/{Path(chunker_cfg.output_path).stem}"
        )

        chunker_info = {
            "chunker_name": chunker_cfg.chunker,
            "chunker_params": chunker_cfg.chunker_arguments,
        }

        pusher.push_dataset(
            hub_dataset_id=repository_id,
            knowledgebase_path=chunker_cfg.output_path,
            chunker_info=chunker_info,
            private=chunker_cfg.upload_config.hub_private,
        )
        logger.info("Successfully uploaded chunks to Hub: %s", repository_id)
    except Exception as exc:
        logger.error("Failed to upload chunks to Hub: %s", exc)


def generate_questions(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate question-answer pairs and optionally upload to the Hub."""

    if not config.question_generation:
        raise ValueError("Question generation config is required but not provided")

    question_cfg = config.question_generation

    if not question_cfg.litellm_config:
        raise ValueError("Question generation requires 'litellm_config' settings.")

    litellm_config = question_cfg.litellm_config

    generator = QuestionGenerator(
        prompt=QUESTION_GENERATION_PROMPT,
        llm_model=litellm_config.model,
        embedding_model=litellm_config.embedding_model,
        dedup_enabled=question_cfg.deduplication_enabled,
        similarity_threshold=question_cfg.similarity_threshold,
        max_workers=question_cfg.max_workers,
        model_api_base=litellm_config.model_api_base,
        embedding_api_base=litellm_config.embedding_api_base,
        embedding_batch_size=question_cfg.dedup_embedding_batch_size,
        llm_calls_per_minute=question_cfg.llm_calls_per_minute,
        embedding_calls_per_minute=question_cfg.embedding_calls_per_minute,
        temperature=litellm_config.temperature,
    )

    text_to_chunk_map: Dict[str, List[str]] = {}
    for item in kb_dataset:
        chunk_text = item["text"]
        text_to_chunk_map.setdefault(chunk_text, []).append(item["id"])

    unique_texts = list(text_to_chunk_map.keys())
    logger.info("Found %d unique chunks", len(unique_texts))

    questions, metrics = generator.generate_for_chunks(unique_texts)
    logger.info(
        "Generated %d questions after deduplication", metrics["num_questions_deduped"]
    )
    logger.info("Question generation metrics: %s", metrics)

    train_records: List[Dict[str, Any]] = []
    skipped_questions = 0

    for question in questions:
        chunk_text = question.get("chunk_text")
        if not chunk_text:
            skipped_questions += 1
            continue

        chunk_ids = text_to_chunk_map.get(chunk_text, [])
        if not chunk_ids:
            skipped_questions += 1
            logger.warning(
                "Could not find chunk_id for question: %s...",
                question["question"][:100],
            )
            continue

        for chunk_id in chunk_ids:
            train_records.append(
                {
                    "anchor": question["question"],
                    "positive": chunk_text,
                    "question_id": question["id"],
                    "chunk_id": chunk_id,
                }
            )

    logger.info(
        "Created %d training records (skipped %d questions)",
        len(train_records),
        skipped_questions,
    )

    if question_cfg.output_path:
        output_path = Path(question_cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(train_records, file, indent=2, ensure_ascii=False)
            logger.info("Saved training records to %s", output_path)
        except Exception as exc:
            logger.error("Failed to save training records: %s", exc)

    if (
        config.hub_username
        and question_cfg.upload_config
        and question_cfg.upload_config.push_to_hub
    ):
        _push_questions_to_hub(config, metrics)

    return train_records, metrics


def _push_questions_to_hub(config: PipelineConfig, metrics: Dict[str, int]) -> None:
    question_cfg = config.question_generation
    if not question_cfg:
        return

    if not config.hub_username:
        logger.warning("Skipping question dataset upload: no Hugging Face username configured.")
        return

    token = config.hub_token or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("Skipping question dataset upload: no Hugging Face token available.")
        return

    try:
        pusher = DatasetPusher(username=config.hub_username, token=token)

        repository_id = question_cfg.upload_config.hub_dataset_id

        question_gen_info = {
            "model_name": question_cfg.litellm_config.model,
            "similarity_threshold": question_cfg.similarity_threshold,
            "num_questions": metrics.get("num_questions_original"),
            "num_deduped": metrics.get("num_questions_deduped"),
        }

        pusher.push_dataset(
            hub_dataset_id=repository_id,
            train_path=question_cfg.output_path,
            question_gen_info=question_gen_info,
            private=question_cfg.upload_config.hub_private,
        )
        logger.info("Successfully uploaded train dataset to Hub: %s", repository_id)
    except Exception as exc:
        logger.error("Failed to upload train dataset to Hub: %s", exc)


def train_model(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
    train_dataset: List[Dict[str, Any]],
) -> None:
    """Train the embedding model."""

    train_main(config, train_dataset=train_dataset, kb_dataset=kb_dataset)