import os
import json
import logging
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

def load_main_config(config_path_or_obj):
    """Load configuration from path or use provided config object."""
    if isinstance(config_path_or_obj, (str, bytes, os.PathLike)):
        with open(config_path_or_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return config_path_or_obj.model_dump()

def validate_dataset_consistency(kb_dataset: List[Dict[str, Any]], train_dataset: List[Dict[str, Any]]) -> bool:
    """Validate that all chunk IDs referenced in training data exist in knowledgebase."""
    logger.info("Validating dataset consistency...")
    
    # Get all chunk IDs from knowledgebase
    kb_ids = {str(item['id']) for item in kb_dataset}
    logger.info(f"Found {len(kb_ids)} unique chunks in knowledgebase")
    
    # Get all chunk IDs referenced in training data
    train_chunk_ids = {str(item['chunk_id']) for item in train_dataset}
    logger.info(f"Found {len(train_chunk_ids)} unique chunk references in training data")
    
    # Find missing chunks
    missing_chunks = train_chunk_ids - kb_ids
    if missing_chunks:
        logger.error(f"Found {len(missing_chunks)} chunk IDs in training data that don't exist in knowledgebase")
        logger.error(f"First few missing IDs: {list(missing_chunks)[:5]}")
        return False
        
    logger.info("Dataset consistency validation passed!")
    return True

def build_evaluation_structures(kb_dataset, test_dataset, kb_id_field="id", kb_text_field="text"):
    corpus = {row[kb_id_field]: row[kb_text_field] for row in kb_dataset}
    queries = {row["id"]: row["anchor"] for row in test_dataset}
    relevant_docs = {}
    for row in test_dataset:
        q_id = row["id"]
        if "global_chunk_id" not in row:
            logger.warning(f"Missing 'global_chunk_id': {row}")
            continue
        if q_id not in relevant_docs:
            relevant_docs[q_id] = []
        relevant_docs[q_id].append(row["global_chunk_id"])
    return corpus, queries, relevant_docs

def format_evaluation_results(title: str, results: dict, dim_list: list, metrics: list = None):
    if metrics is None:
        metrics = [
            "ndcg@10", "mrr@10", "map@100",
            "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
            "precision@1", "precision@3", "precision@5", "precision@10",
            "recall@1", "recall@3", "recall@5", "recall@10",
        ]

    # Calculate required widths for alignment
    max_dim_length = max(len(str(dim)) for dim in dim_list)
    value_format_length = 5  # '0.000' is 5 characters
    column_width = max(max_dim_length, value_format_length)
    metric_width = max(len(metric) for metric in metrics) if metrics else 10

    # Prepare dimension headers (left-aligned strings)
    dim_header = "  ".join(f"{str(dim):<{column_width}}" for dim in dim_list)
    
    # Create header line and dynamically determine separator length
    header_line = f"{'Metric':{metric_width}}  {dim_header}"
    separator = "-" * len(header_line)

    output = [
        f"\n{title}",
        separator,
        header_line,
        separator
    ]

    # Populate each metric row (values right-aligned)
    for metric in metrics:
        values = []
        for dim in dim_list:
            key = f"dim_{dim}_cosine_{metric}"
            val = results.get(key, 0.0)
            values.append(f"{val:>{column_width}.3f}")  # Right-align values
        metric_line = f"{metric:{metric_width}}  {'  '.join(values)}"
        output.append(metric_line)
    
    # Final row
    output.append(separator)
    
    return "\n".join(output)

def run_baseline_eval(model, evaluator, dim_list):
    results = evaluator(model)
    print(format_evaluation_results("Before Training (Baseline) Results", results, dim_list))
    return results

def run_final_eval(model, evaluator, dim_list):
    results = evaluator(model)
    print(format_evaluation_results("After Training (Fine-Tuned) Results", results, dim_list))
    return results

def save_metrics_to_file(before: dict, after: dict, dim_list: list, path="metrics_comparison.txt"):
    metrics = [
        "ndcg@10", "mrr@10", "map@100",
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
    ]

    def format_dimensions(dim_list):
        return [f"{dim:,}" for dim in dim_list]

    def write_table(f, title, headers, rows, value_formatter):
        # Calculate column widths
        is_percentage = "%" in rows[0][1]
        
        # For metric column
        metric_width = max(len(row[0]) for row in rows)
        metric_width = max(metric_width, len(headers[0]))
        
        # For value columns, ensure we account for the maximum width including +/- and %
        dim_widths = []
        for col_idx in range(1, len(headers)):
            col_values = [row[col_idx] for row in rows]
            max_width = max(len(str(val)) for val in col_values)
            header_width = len(headers[col_idx])
            dim_widths.append(max(max_width, header_width))
        
        col_widths = [metric_width] + dim_widths
        
        # Build header with consistent spacing
        header = headers[0].ljust(col_widths[0])
        for i, h in enumerate(headers[1:], 1):
            header += " │ " + h.center(col_widths[i])
        
        # Build separator
        separator = "─" * col_widths[0]
        for w in col_widths[1:]:
            separator += "─┼─" + "─" * w
            
        # Write table
        f.write(f"\n{title}\n")
        f.write("-" * len(title) + "\n")
        f.write(f"{header}\n{separator}\n")
        
        # Write rows with consistent spacing
        for row in rows:
            line = row[0].ljust(col_widths[0])
            for i, val in enumerate(row[1:], 1):
                if is_percentage:
                    # Right-align percentage values
                    line += " │ " + val.rjust(col_widths[i])
                else:
                    line += " │ " + value_formatter(val, col_widths[i])
            f.write(line + "\n")

    with open(path, "w", encoding="utf-8") as f:
        # Main title
        f.write("Model Performance Metrics Comparison\n")
        f.write("====================================\n")
        
        # Format dimensions with thousands separators
        dim_strs = format_dimensions(dim_list)
        
        # Prepare data tables
        baseline_rows = []
        finetuned_rows = []
        delta_rows = []
        pct_change_rows = []
        
        for metric in metrics:
            bl_vals = []
            ft_vals = []
            dt_vals = []
            pc_vals = []
            
            for dim in dim_list:
                key = f"dim_{dim}_cosine_{metric}"
                b_val = before.get(key, 0.0)
                a_val = after.get(key, 0.0)
                delta = a_val - b_val
                pct_change = (delta / b_val * 100) if b_val != 0 else 0.0
                
                bl_vals.append(f"{b_val:.3f}")
                ft_vals.append(f"{a_val:.3f}")
                dt_vals.append(f"{delta:+.3f}")
                pc_vals.append(f"{pct_change:+.1f}%")
            
            baseline_rows.append([metric] + bl_vals)
            finetuned_rows.append([metric] + ft_vals)
            delta_rows.append([metric] + dt_vals)
            pct_change_rows.append([metric] + pc_vals)

        # Write tables
        headers = ["Metric"] + dim_strs
        num_formatter = lambda val, w: f"{val:>{w}}"
        
        write_table(f, "Baseline Performance", headers, baseline_rows, num_formatter)
        write_table(f, "Fine-Tuned Performance", headers, finetuned_rows, num_formatter)
        write_table(f, "Absolute Changes (Δ)", headers, delta_rows, num_formatter)
        write_table(f, "Percentage Changes", headers, pct_change_rows, num_formatter)

def select_device(preferred_device: Optional[str] = None) -> str:
    # Handle explicit preference first
    if preferred_device:
        if preferred_device == "cuda" and torch.cuda.is_available():
            logger.info("Using CUDA GPU as requested")
            return "cuda"
        elif preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS as requested")
            return "mps"
        elif preferred_device == "cpu":
            logger.info("Using CPU as requested")
            return "cpu"
        else:
            logger.warning(f"Requested device '{preferred_device}' not available, falling back to auto-detection")
    
    # Auto-detection (prioritize GPU > MPS > CPU)
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected and selected")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS detected and selected")
        return "mps"
    else:
        logger.info("No accelerator detected, using CPU")
        return "cpu"

def main(config, train_dataset: List[Dict[str, Any]], kb_dataset: List[Dict[str, Any]]):
    """Main training function."""
    if not hasattr(config, 'training') or not config.training:
        raise ValueError("Training configuration is required but not provided")

    # Validate dataset consistency before proceeding
    if not validate_dataset_consistency(kb_dataset, train_dataset):
        raise ValueError("Dataset consistency check failed - chunks and training data are mismatched. "
                        "This usually happens when chunks have been regenerated after question generation.")

    # Convert lists to dataset objects if needed
    train_dataset_full = Dataset.from_list(train_dataset)

    if "id" not in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.add_column("id", list(range(len(train_dataset_full))))
    if "chunk_id" in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.rename_column("chunk_id", "global_chunk_id")
    train_dataset_full = train_dataset_full.shuffle()
    dataset_split = train_dataset_full.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    logger.info(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    # Build evaluation structures
    corpus, queries, relevant_docs = build_evaluation_structures(
        kb_dataset=kb_dataset,
        test_dataset=test_dataset,
        kb_id_field="id",
        kb_text_field="text"
    )

    # Setup evaluators
    dim_list = config.training.model_settings.matryoshka_dimensions
    evaluators = []
    for d in dim_list:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{d}",
                score_functions={"cosine": cos_sim},
                truncate_dim=d
            )
        )
    evaluator = SequentialEvaluator(evaluators)

    # Initialize base model and run baseline evaluation
    device = select_device(config.training.training_arguments.device)
    base_model = SentenceTransformer(config.training.model_settings.model_id, device=device, trust_remote_code=config.training.model_settings.trust_remote_code)
    base_model.max_seq_length = config.training.model_settings.max_seq_length
    baseline_results = run_baseline_eval(base_model, evaluator, dim_list)

    logger.info("Re-initializing for training.")
    model = SentenceTransformer(
        config.training.model_settings.model_id,
        device=device,
        trust_remote_code=config.training.model_settings.trust_remote_code,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Fine-tuned with [QuicKB](https://github.com/ALucek/QuicKB)",
        ),
    )
    model.max_seq_length = config.training.model_settings.max_seq_length

    # Setup loss functions
    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=dim_list
    )

    # Setup training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=config.training.training_arguments.output_path,
        num_train_epochs=config.training.training_arguments.epochs,
        per_device_train_batch_size=config.training.training_arguments.batch_size,
        gradient_accumulation_steps=config.training.training_arguments.gradient_accumulation_steps,
        learning_rate=config.training.training_arguments.learning_rate,
        warmup_ratio=config.training.training_arguments.warmup_ratio,
        lr_scheduler_type=config.training.training_arguments.lr_scheduler_type,
        optim=config.training.training_arguments.optim,
        tf32=config.training.training_arguments.tf32,
        bf16=config.training.training_arguments.bf16,
        batch_sampler=config.training.training_arguments.batch_sampler.value,
        eval_strategy=config.training.training_arguments.eval_strategy,
        save_strategy=config.training.training_arguments.save_strategy,
        logging_steps=config.training.training_arguments.logging_steps,
        save_total_limit=config.training.training_arguments.save_total_limit,
        load_best_model_at_end=config.training.training_arguments.load_best_model_at_end,
        metric_for_best_model=config.training.model_settings.metric_for_best_model,
        report_to=config.training.training_arguments.report_to,
    )

    # Prepare final training dataset and trainer
    final_train_dataset = train_dataset.select_columns(["anchor", "positive"])
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=final_train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    
    # Evaluate fine-tuned model
    fine_tuned_model = SentenceTransformer(config.training.training_arguments.output_path, device=device, trust_remote_code=config.training.model_settings.trust_remote_code)
    fine_tuned_model.max_seq_length = config.training.model_settings.max_seq_length
    final_results = run_final_eval(fine_tuned_model, evaluator, dim_list)

    # Save metrics
    save_metrics_to_file(
        baseline_results, 
        final_results, 
        dim_list, 
        path=f"{config.training.training_arguments.output_path}/metrics_comparison.txt"
    )

    # Handle model upload if configured
    if (config.training.upload_config and 
        config.training.upload_config.push_to_hub):
            
        if not config.hub_token and not os.getenv("HF_TOKEN"):
            logger.warning("No HF_TOKEN in env or config, attempting login anyway.")
                
        if not config.training.upload_config.hub_model_id:
            logger.warning("No hub_model_id specified, skipping upload")
        else:
            logger.info(f"Pushing model to HF Hub: {config.training.upload_config.hub_model_id}")
            trainer.model.push_to_hub(
                config.training.upload_config.hub_model_id, 
                exist_ok=True, 
                private=config.training.upload_config.hub_private
            )
            logger.info("Upload complete!")

    logger.info("Training pipeline finished.")
