import logging
import os
import json
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from datasets import load_dataset, Dataset

from chunking import ChunkerRegistry
from hub_upload.dataset_pusher import DatasetPusher
from synth_dataset.question_generator import QuestionGenerator
from training.train import main as train_main

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class PipelineStage(Enum):
    CHUNK = auto()
    GENERATE = auto()
    TRAIN = auto()

class BatchSamplers(str, Enum):
    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"

class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM model and embedding settings."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    model: Optional[str] = "openai/gpt-4o"
    model_api_base: Optional[str] = None
    embedding_model: Optional[str] = "openai/text-embedding-3-large"
    embedding_api_base: Optional[str] = None

class QuestionGenInputConfig(BaseModel):
    """Configuration for question generation input dataset."""
    model_config = ConfigDict(extra='forbid', validate_default=True)

    dataset_source: Literal["local", "hub"] = "local"
    knowledgebase_dataset_id: Optional[str] = None
    local_knowledgebase_path: Optional[str] = None

class TrainInputConfig(BaseModel):
    """Configuration for training input datasets."""
    model_config = ConfigDict(extra='forbid', validate_default=True)

    dataset_source: Literal["local", "hub"] = "local"
    train_dataset_id: Optional[str] = None
    knowledgebase_dataset_id: Optional[str] = None
    local_train_path: Optional[str] = None
    local_knowledgebase_path: Optional[str] = None

class UploadConfig(BaseModel):
    """Configuration for Hugging Face Hub uploads."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    push_to_hub: bool = False
    hub_private: bool = False
    hub_dataset_id: Optional[str] = None
    hub_model_id: Optional[str] = None

class ChunkerConfig(BaseModel):
    """Configuration for text chunking."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    chunker: str
    chunker_arguments: Dict[str, Any]
    output_path: str
    upload_config: Optional[UploadConfig] = None

    @property
    def litellm_config(self) -> Optional[LiteLLMConfig]:
        """Extract ModelConfig from chunker_arguments if present."""
        if "litellm_config" in self.chunker_arguments:
            return LiteLLMConfig.model_validate(self.chunker_arguments["litellm_config"])
        return None

class QuestionGeneratorConfig(BaseModel):
    """Configuration for question generation."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
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
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    model_id: str
    matryoshka_dimensions: List[int] = [768, 512, 256, 128, 64]
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10"
    max_seq_length: Optional[int] = 768
    trust_remote_code: Optional[bool] = False

class TrainingArguments(BaseModel):
    """Arguments for model training."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    # Required parameters
    output_path: str
    device: Optional[str] = "cuda"
    
    # Basic training parameters
    epochs: Optional[int] = 4
    batch_size: Optional[int] = 32
    gradient_accumulation_steps: Optional[int] = 16
    learning_rate: Optional[float] = 2.0e-5
    
    # Learning rate scheduler settings
    warmup_ratio: Optional[float] = 0.1
    lr_scheduler_type: Optional[str] = "cosine"
    
    # Optimizer settings
    optim: Optional[str] = "adamw_torch_fused"
    
    # Hardware optimization flags
    tf32: Optional[bool] = True
    bf16: Optional[bool] = True
    
    # Batch sampling strategy
    batch_sampler: Optional[BatchSamplers] = BatchSamplers.NO_DUPLICATES
    
    # Training and evaluation strategy
    eval_strategy: Optional[str] = "epoch"
    save_strategy: Optional[str] = "epoch"
    logging_steps: Optional[int] = 10
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: Optional[bool] = True
    
    # Reporting
    report_to: Optional[str] = "none"

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    model_settings: ModelSettings
    training_arguments: TrainingArguments
    train_dataset_config: TrainInputConfig
    upload_config: Optional[UploadConfig] = None

class PipelineConfig(BaseModel):
    """Main configuration for the QuicKB pipeline."""
    model_config = ConfigDict(extra='forbid', validate_default=True)
    
    pipeline: Dict[str, str]
    hub_username: Optional[str] = None
    hub_token: Optional[str] = None
    path_to_knowledgebase: Optional[str]
    chunker_config: Optional[ChunkerConfig] = None
    question_generation: Optional[QuestionGeneratorConfig] = None
    training: Optional[TrainingConfig] = None

def load_dataset_from_local(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a local JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in {file_path}")
        raise

def load_dataset_from_hub(hub_dataset_id: str) -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face Hub using default config."""
    try:
        logger.info(f"Loading dataset from Hub: {hub_dataset_id}")
        dataset = load_dataset(hub_dataset_id, split="train")
        if dataset:
            return dataset.to_list()
        else:
            logger.error(f"No data found in dataset: {hub_dataset_id}")
            return []
    except Exception as e:
        logger.error(f"Error loading dataset from Hub: {hub_dataset_id}. Error: {e}")
        raise


def load_pipeline_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return PipelineConfig.model_validate(config_data)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def process_chunks(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Process documents into chunks and optionally upload to Hub."""
    
    # Get chunker class
    chunker_class = ChunkerRegistry.get_chunker(config.chunker_config.chunker)
    args = config.chunker_config.chunker_arguments.copy()
    chunker = chunker_class(**args)

    logger.info(f"Initialized Chunker: {config.chunker_config.chunker}")
    
    # Process files
    base_path = Path(config.path_to_knowledgebase)
    results = []
    total_chunks = 0
    
    for file_path in base_path.rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunker.split_text(text)
            source_path = str(file_path.relative_to(base_path))
            
            for chunk in chunks:
                results.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": source_path
                })
            
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            total_chunks += len(chunks)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    logger.info(f"Created {total_chunks} chunks in total")
    
    # Save results
    output_path = Path(config.chunker_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Handle upload if configured
    if (config.hub_username and 
        config.chunker_config.upload_config and 
        config.chunker_config.upload_config.push_to_hub):
        try:
            pusher = DatasetPusher(
                username=config.hub_username,
                token=config.hub_token
            )
            
            repository_id = (config.chunker_config.upload_config.hub_dataset_id 
                           or f"{config.hub_username}/{Path(config.chunker_config.output_path).stem}")
            
            chunker_info = {
                'chunker_name': config.chunker_config.chunker,
                'chunker_params': config.chunker_config.chunker_arguments
            }
            
            pusher.push_dataset(
                hub_dataset_id=repository_id,
                knowledgebase_path=config.chunker_config.output_path,
                chunker_info=chunker_info,
                private=config.chunker_config.upload_config.hub_private
            )
            logger.info(f"Successfully uploaded chunks to Hub: {repository_id}")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Hub: {str(e)}")
    
    return results

def generate_questions(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate questions and optionally upload to Hub."""

    if not config.question_generation:
        raise ValueError("Question generation config is required but not provided")
    
    if not config.question_generation.litellm_config:
        raise ValueError("Question generation requires 'litellm_config' settings.")

    litellm_config = config.question_generation.litellm_config

    generator = QuestionGenerator(
        prompt_path="src/prompts/question_generation.txt",
        llm_model=litellm_config.model,
        embedding_model=litellm_config.embedding_model,
        dedup_enabled=config.question_generation.deduplication_enabled,
        similarity_threshold=config.question_generation.similarity_threshold,
        max_workers=config.question_generation.max_workers,
        model_api_base=litellm_config.model_api_base,
        embedding_api_base=litellm_config.embedding_api_base,
        embedding_batch_size=config.question_generation.dedup_embedding_batch_size,
        llm_calls_per_minute=config.question_generation.llm_calls_per_minute,
        embedding_calls_per_minute=config.question_generation.embedding_calls_per_minute
    )
    
    # Get unique texts and build a text-to-id mapping
    text_to_chunk_map = {}
    for item in kb_dataset:
        text_val = item["text"]
        if text_val not in text_to_chunk_map:
            text_to_chunk_map[text_val] = []
        text_to_chunk_map[text_val].append(item["id"])
    
    unique_texts = list(text_to_chunk_map.keys())
    logger.info(f"Found {len(unique_texts)} unique chunks")
    
    # Generate questions
    questions = generator.generate_for_chunks(unique_texts)
    logger.info(f"Generated {len(questions)} questions after deduplication")
    
    # Track metrics
    metrics = {
        "num_questions_original": sum(len(generator._question_cache[chunk]) for chunk in generator._question_cache),
        "num_questions_deduped": len(questions)
    }
    logger.info(f"Question generation metrics: {metrics}")
    
    # Create training records
    train_records = []
    skipped_questions = 0
    for q in questions:
        chunk_text = q.get("chunk_text")
        if not chunk_text:
            skipped_questions += 1
            continue
            
        chunk_ids = text_to_chunk_map.get(chunk_text, [])
        if not chunk_ids:
            skipped_questions += 1
            logger.warning(f"Could not find chunk_id for question: {q['question'][:100]}...")
            continue
            
        # Create a record for each matching chunk
        for chunk_id in chunk_ids:
            train_records.append({
                "anchor": q["question"],
                "positive": chunk_text,
                "question_id": q["id"],
                "chunk_id": chunk_id
            })
    
    logger.info(f"Created {len(train_records)} training records (skipped {skipped_questions} questions)")
    
    # Save results
    if config.question_generation.output_path:
        output_path = Path(config.question_generation.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(train_records, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved training records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training records: {str(e)}")

    # Handle upload if configured
    if (config.hub_username and 
        config.question_generation.upload_config and 
        config.question_generation.upload_config.push_to_hub):
        try:
            pusher = DatasetPusher(
                username=config.hub_username,
                token=config.hub_token
            )
            
            repository_id = config.question_generation.upload_config.hub_dataset_id
            
            question_gen_info = {
                'model_name': config.question_generation.litellm_config.model,
                'similarity_threshold': config.question_generation.similarity_threshold,
                'num_questions': metrics['num_questions_original'],
                'num_deduped': metrics['num_questions_deduped']
            }
            
            pusher.push_dataset(
                hub_dataset_id=repository_id,
                train_path=config.question_generation.output_path,
                question_gen_info=question_gen_info,
                private=config.question_generation.upload_config.hub_private
            )
            logger.info(f"Successfully uploaded train dataset to Hub: {repository_id}")
        except Exception as e:
            logger.error(f"Failed to upload train dataset to Hub: {str(e)}")
    
    return train_records, metrics

def train_model(config: PipelineConfig, kb_dataset: List[Dict[str, Any]], train_dataset: List[Dict[str, Any]]):
    """Train the embedding model."""
    train_main(config, train_dataset=train_dataset, kb_dataset=kb_dataset)

def upload_to_hub(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
    train_dataset: Optional[List[Dict[str, Any]]] = None,
    question_metrics: Optional[Dict[str, int]] = None
):
    """Upload datasets to Hugging Face Hub."""
    
    if not config.hub_username:
        logger.warning("No 'hub_username' specified, skipping upload.")
        return
        
    try:
        # Initialize pusher
        pusher = DatasetPusher(
            username=config.hub_username,
            token=config.hub_token
        )
        
        # Get repository name from output path
        repository_name = Path(config.output_path).stem
        
        # Collect chunker info
        chunker_info = {
            'chunker_name': config.chunker,
            'chunker_params': config.chunker_arguments
        }
        
        # Collect question generation info if enabled
        question_gen_info = None
        question_gen_info = {
            'model_name': config.question_generation.model,
            'similarity_threshold': config.deduplication.similarity_threshold,
            'num_questions': question_metrics['num_questions_original'] if question_metrics else len(train_dataset),
            'num_deduped': question_metrics['num_questions_deduped'] if question_metrics else len(train_dataset)
        }
    
        # Push dataset
        pusher.push_dataset(
            repository_name=repository_name,
            knowledgebase_path=config.output_path,
            chunker_info=chunker_info,
            train_path=config.question_output_path if train_dataset else None,
            question_gen_info=question_gen_info,
            private=config.hub_private
        )
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face Hub: {str(e)}")

def run_pipeline(config: PipelineConfig):
    """Run the QuicKB pipeline."""
    from_stage = PipelineStage[config.pipeline["from_stage"]]
    to_stage = PipelineStage[config.pipeline["to_stage"]]

    kb_dataset = None
    train_dataset = None
    question_metrics = None

    # 1. CHUNK
    if from_stage.value <= PipelineStage.CHUNK.value <= to_stage.value:
        logger.info("Running CHUNK stage.")
        kb_dataset = process_chunks(config)
    else:
        logger.info("Skipping CHUNK stage.")

    # 2. GENERATE
    if from_stage.value <= PipelineStage.GENERATE.value <= to_stage.value:
        # Load knowledgebase dataset if needed for GENERATE
        if kb_dataset is None:
            input_config = config.question_generation.input_dataset_config
            if input_config.dataset_source == "hub":
                logger.info(f"Loading knowledgebase dataset from Hub: {input_config.knowledgebase_dataset_id}")
                kb_dataset = load_dataset_from_hub(input_config.knowledgebase_dataset_id)
            elif input_config.dataset_source == "local":
                local_kb_path = input_config.local_knowledgebase_path or config.chunker_config.output_path
                logger.info(f"Loading knowledgebase dataset from local path: {local_kb_path}")
                kb_dataset = load_dataset_from_local(local_kb_path)

        logger.info("Running GENERATE stage.")
        train_dataset, question_metrics = generate_questions(config, kb_dataset)

    # 3. TRAIN
    if from_stage.value <= PipelineStage.TRAIN.value <= to_stage.value:
        logger.info("Running TRAIN stage.")
        if not config.training:
            raise ValueError("No training config found, cannot run TRAIN stage.")

        train_config = config.training.train_dataset_config
        
        # Load datasets for training if needed
        if train_config.dataset_source == "hub":
            logger.info("Loading datasets from Hub for training...")
            if train_dataset is None:
                logger.info(f"Loading training dataset from Hub: {train_config.train_dataset_id}")
                train_dataset = load_dataset_from_hub(train_config.train_dataset_id)
            if kb_dataset is None:
                logger.info(f"Loading knowledgebase dataset from Hub: {train_config.knowledgebase_dataset_id}")
                kb_dataset = load_dataset_from_hub(train_config.knowledgebase_dataset_id)
        else:  # local
            logger.info("Loading datasets from local files for training...")
            if train_dataset is None:
                train_path = train_config.local_train_path or config.question_generation.output_path
                logger.info(f"Loading training dataset from local path: {train_path}")
                train_dataset = load_dataset_from_local(train_path)
            if kb_dataset is None:
                kb_path = train_config.local_knowledgebase_path or config.chunker_config.output_path
                logger.info(f"Loading knowledgebase dataset from local path: {kb_path}")
                kb_dataset = load_dataset_from_local(kb_path)

        if not kb_dataset or not train_dataset:
            raise ValueError("Failed to load required datasets for training")

        train_model(config, kb_dataset, train_dataset)

    logger.info("Pipeline complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_pipeline_config("config.yaml")
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
