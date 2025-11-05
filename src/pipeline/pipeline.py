from __future__ import annotations

from enum import Enum, auto
import logging
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .datasets import load_dataset_from_hub, load_dataset_from_local
from .stages import ChunkGenerationResult, generate_questions, process_chunks, train_model

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    CHUNK = auto()
    GENERATE = auto()
    TRAIN = auto()


def run_pipeline(config: PipelineConfig) -> None:
    """Run the configured pipeline stages."""

    from_stage = PipelineStage[config.pipeline["from_stage"]]
    to_stage = PipelineStage[config.pipeline["to_stage"]]

    chunk_result: Optional[ChunkGenerationResult] = None
    kb_dataset: Optional[List[Dict[str, Any]]] = None
    train_dataset: Optional[List[Dict[str, Any]]] = None
    question_metrics: Optional[Dict[str, int]] = None

    if from_stage.value <= PipelineStage.CHUNK.value <= to_stage.value:
        logger.info("Running CHUNK stage.")
        chunk_result = process_chunks(config)
    else:
        logger.info("Skipping CHUNK stage.")

    if from_stage.value <= PipelineStage.GENERATE.value <= to_stage.value:
        if not config.question_generation:
            raise ValueError(
                "Question generation config is required when running GENERATE stage."
            )

        created_from_chunk_output = False

        if kb_dataset is None:
            if chunk_result:
                kb_dataset = list(chunk_result.iter_chunks())
                created_from_chunk_output = True
            else:
                input_config = config.question_generation.input_dataset_config

                if input_config.dataset_source == "hub":
                    if not input_config.knowledgebase_dataset_id:
                        raise ValueError("knowledgebase_dataset_id is required for Hub datasets")
                    logger.info(
                        "Loading knowledgebase dataset from Hub: %s",
                        input_config.knowledgebase_dataset_id,
                    )
                    kb_dataset = load_dataset_from_hub(input_config.knowledgebase_dataset_id)
                else:
                    local_kb_path = input_config.local_knowledgebase_path
                    if not local_kb_path:
                        if not config.chunker_config:
                            raise ValueError(
                                "chunker_config is required to locate local knowledgebase data"
                            )
                        local_kb_path = config.chunker_config.output_path
                    logger.info("Loading knowledgebase dataset from local path: %s", local_kb_path)
                    kb_dataset = load_dataset_from_local(local_kb_path)

        logger.info("Running GENERATE stage.")
        train_dataset, question_metrics = generate_questions(config, kb_dataset)

        # Release memory if we can reconstruct from disk later
        if created_from_chunk_output:
            kb_dataset = None

    if from_stage.value <= PipelineStage.TRAIN.value <= to_stage.value:
        logger.info("Running TRAIN stage.")

        if not config.training:
            raise ValueError("No training config found, cannot run TRAIN stage.")

        train_config = config.training.train_dataset_config

        if train_dataset is None:
            if train_config.dataset_source == "hub":
                if not train_config.train_dataset_id:
                    raise ValueError("train_dataset_id is required for Hub datasets")
                logger.info(
                    "Loading training dataset from Hub: %s",
                    train_config.train_dataset_id,
                )
                train_dataset = load_dataset_from_hub(train_config.train_dataset_id)
            else:
                local_train_path = train_config.local_train_path
                if not local_train_path:
                    if not config.question_generation:
                        raise ValueError(
                            "question_generation config is required to locate local train data"
                        )
                    local_train_path = config.question_generation.output_path
                logger.info("Loading training dataset from local path: %s", local_train_path)
                train_dataset = load_dataset_from_local(local_train_path)

        if kb_dataset is None:
            if chunk_result:
                kb_dataset = list(chunk_result.iter_chunks())
            else:
                if train_config.dataset_source == "hub":
                    if not train_config.knowledgebase_dataset_id:
                        raise ValueError("knowledgebase_dataset_id is required for Hub datasets")
                    logger.info(
                        "Loading knowledgebase dataset from Hub: %s",
                        train_config.knowledgebase_dataset_id,
                    )
                    kb_dataset = load_dataset_from_hub(train_config.knowledgebase_dataset_id)
                else:
                    kb_path = train_config.local_knowledgebase_path
                    if not kb_path:
                        if not config.chunker_config:
                            raise ValueError(
                                "chunker_config is required to locate local knowledgebase data"
                            )
                        kb_path = config.chunker_config.output_path
                    logger.info("Loading knowledgebase dataset from local path: %s", kb_path)
                    kb_dataset = load_dataset_from_local(kb_path)

        if not kb_dataset or not train_dataset:
            raise ValueError("Failed to load required datasets for training")

        train_model(config, kb_dataset, train_dataset)

    logger.info("Pipeline complete!")


