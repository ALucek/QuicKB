"""Configuration models and loaders for the QuicKB pipeline."""

from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class BatchSamplers(str, Enum):
    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"


class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM model and embedding settings."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model: Optional[str] = "openai/gpt-4o"
    model_api_base: Optional[str] = None
    embedding_model: Optional[str] = "openai/text-embedding-3-large"
    embedding_api_base: Optional[str] = None
    temperature: Optional[float] = None


class QuestionGenInputConfig(BaseModel):
    """Configuration for question generation input dataset."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    dataset_source: Literal["local", "hub"] = "local"
    knowledgebase_dataset_id: Optional[str] = None
    local_knowledgebase_path: Optional[str] = None


class TrainInputConfig(BaseModel):
    """Configuration for training input datasets."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    dataset_source: Literal["local", "hub"] = "local"
    train_dataset_id: Optional[str] = None
    knowledgebase_dataset_id: Optional[str] = None
    local_train_path: Optional[str] = None
    local_knowledgebase_path: Optional[str] = None


class UploadConfig(BaseModel):
    """Configuration for Hugging Face Hub uploads."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    push_to_hub: bool = False
    hub_private: bool = False
    hub_dataset_id: Optional[str] = None
    hub_model_id: Optional[str] = None


class ChunkerConfig(BaseModel):
    """Configuration for text chunking."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    chunker: str
    chunker_arguments: Dict[str, Any]
    output_path: str
    upload_config: Optional[UploadConfig] = None

    @property
    def litellm_config(self) -> Optional[LiteLLMConfig]:  # pragma: no cover - convenience accessor
        if "litellm_config" in self.chunker_arguments:
            return LiteLLMConfig.model_validate(self.chunker_arguments["litellm_config"])
        return None


class QuestionGeneratorConfig(BaseModel):
    """Configuration for question generation."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    output_path: str
    input_dataset_config: QuestionGenInputConfig
    litellm_config: Optional[LiteLLMConfig]
    max_workers: Optional[int] = 20
    deduplication_enabled: Optional[bool] = True
    dedup_embedding_batch_size: Optional[int] = 500
    similarity_threshold: Optional[float] = 0.85
    upload_config: Optional[UploadConfig] = None
    llm_calls_per_minute: Optional[int] = 15
    embedding_calls_per_minute: Optional[int] = 15


class ModelSettings(BaseModel):
    """Settings for the embedding model training."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_id: str
    matryoshka_dimensions: List[int] = [768, 512, 256, 128, 64]
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10"
    max_seq_length: Optional[int] = 768
    trust_remote_code: Optional[bool] = False


class TrainingArguments(BaseModel):
    """Arguments for model training."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    output_path: str
    device: Optional[str] = "cuda"
    epochs: Optional[int] = 4
    batch_size: Optional[int] = 32
    gradient_accumulation_steps: Optional[int] = 16
    learning_rate: Optional[float] = 2.0e-5
    warmup_ratio: Optional[float] = 0.1
    lr_scheduler_type: Optional[str] = "cosine"
    optim: Optional[str] = "adamw_torch_fused"
    tf32: Optional[bool] = True
    bf16: Optional[bool] = True
    batch_sampler: Optional[BatchSamplers] = BatchSamplers.NO_DUPLICATES
    eval_strategy: Optional[str] = "epoch"
    save_strategy: Optional[str] = "epoch"
    logging_steps: Optional[int] = 10
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: Optional[bool] = True
    report_to: Optional[str] = "none"


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_settings: ModelSettings
    training_arguments: TrainingArguments
    train_dataset_config: TrainInputConfig
    upload_config: Optional[UploadConfig] = None


class PipelineConfig(BaseModel):
    """Main configuration for the QuicKB pipeline."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    pipeline: Dict[str, str]
    hub_username: Optional[str] = None
    hub_token: Optional[str] = None
    path_to_knowledgebase: Optional[str]
    chunker_config: Optional[ChunkerConfig] = None
    question_generation: Optional[QuestionGeneratorConfig] = None
    training: Optional[TrainingConfig] = None


def load_pipeline_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration from disk."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
    except Exception as exc:  # pragma: no cover - simple I/O wrapper
        logger.error("Error reading config from %s: %s", config_path, exc)
        raise

    try:
        return PipelineConfig.model_validate(config_data)
    except Exception as exc:
        logger.error("Error validating config from %s: %s", config_path, exc)
        raise


__all__ = [
    "BatchSamplers",
    "ChunkerConfig",
    "LiteLLMConfig",
    "ModelSettings",
    "PipelineConfig",
    "QuestionGenInputConfig",
    "QuestionGeneratorConfig",
    "TrainInputConfig",
    "TrainingArguments",
    "TrainingConfig",
    "UploadConfig",
    "load_pipeline_config",
]

