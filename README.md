# QuicKB 

<img src="qkb_logo.png" width=175>

Optimize Document Retrieval with Fine-Tuned KnowledgeBases

## Overview

QuicKB optimizes document retrieval by creating fine-tuned knowledgebases through an end-to-end machine learning pipeline that handles document chunking, synthetic data generation, and embedding model training.

## Key Features

### Document Chunking
Implement state-of-the-art chunking strategies based on [ChromaDB's research](https://research.trychroma.com/evaluating-chunking):
- **Semantic Approaches**: 
  - LLM-guided splits for natural language breakpoints
  - Content-aware clustering for thematic coherence
  - Hybrid semantic-token methods for balanced chunking
- **Token/Character-Based Methods**:
  - Recursive chunking with custom separator hierarchies
  - Precise length-based splitting

### Training Data Generation
Automatically create domain-specific training datasets:
- Generate synthetic question-answer pairs from your content
- Intelligent deduplication using semantic similarity
- Parallel processing for large-scale document sets
- Support for local and cloud LLMs via [LiteLLM](https://docs.litellm.ai/docs/)

### Embedding Optimization
Fine-tune embedding models for your specific domain:
- Custom training using [Sentence Transformers](https://sbert.net/)
- Cross-platform training support (CUDA, MPS, CPU)
- Dimension reduction techniques with Matryoshka Representation Learning
- Comprehensive evaluation across multiple metrics
- Detailed performance comparisons with baseline models

## Installation

```bash
git clone https://github.com/ALucek/QuicKB.git
cd QuicKB

python -m venv quickb-env
source quickb-env/bin/activate  # Windows: quickb-env\Scripts\activate

pip install -e .
```

## Usage

1. Prepare your text documents in a directory
2. Configure the pipeline in `config.yaml`
3. Run:
```bash
python src/main.py
```
4. Enjoy!

## Configuration Guide

The pipeline is controlled through a single `config.yaml` file. Here's a complete configuration example:

```yaml
# ========== Pipeline Control ==========
pipeline:
  from_stage: "CHUNK"    # Options: "CHUNK", "GENERATE", "TRAIN"
  to_stage: "TRAIN"      # Run pipeline from from_stage to to_stage

# ========== Global Settings ==========
path_to_knowledgebase: "./knowledgebase"          # Directory containing source documents
hub_username: "AdamLucek"                         # Hugging Face username
hub_token: null                                   # Optional: Use HF_TOKEN env variable instead

# ========== Document Chunking ==========
chunker_config:
  output_path: "./output/knowledgebase.json"
  
  # Chunker Config:
  chunker: "RecursiveTokenChunker"
  
  chunker_arguments:
    chunk_size: 400
    chunk_overlap: 0
    length_type: "character"
    separators: ["\n\n", "\n", ".", "?", "!", " ", ""]
    keep_separator: true
    is_separator_regex: false
  
  # Optional: Push chunks to Hugging Face Hub
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "AdamLucek/quickb-kb"

# ========== Question Generation ==========
question_generation:
  output_path: "./output/train_data.json"

  # LLM/Embedding Configuration
  litellm_config:
    model: "openai/gpt-4o-mini"
    model_api_base: null     # Optional: Custom API endpoint

    embedding_model: "text-embedding-3-large"
    embedding_api_base: null # Optional: Custom embedding endpoint

  # Input dataset settings
  input_dataset_config:
    dataset_source: "local"  # Options: "local", "hub"
    local_knowledgebase_path: "./output/knowledgebase.json"
    # Hub alternative:
    # knowledgebase_dataset_id: "username/quickb-kb"

  # Performance settings
  max_workers: 150                    # Parallel question generation
  llm_calls_per_minute: null          # null = no limit
  embedding_calls_per_minute: null    # null = no limit

  # Question deduplication
  deduplication_enabled: true
  dedup_embedding_batch_size: 2048    # Batch size for embedding calculation
  similarity_threshold: 0.85          # Semantic Similarity Threshold

  # Optional: Push training data to Hub
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "AdamLucek/quickb-qa"

# ========== Model Training ==========
training:
  # Model configuration
  model_settings:
    # Base model:
    model_id: "nomic-ai/modernbert-embed-base"
    
    # Matryoshka dimensions (must be descending)
    matryoshka_dimensions: [768, 512, 256, 128, 64]
    metric_for_best_model: "eval_dim_128_cosine_ndcg@10"
    max_seq_length: 1024
    trust_remote_code: false

  # Training data configuration
  train_dataset_config:
    dataset_source: "local"  # Options: "local", "hub"
    local_train_path: "./output/train_data.json"
    local_knowledgebase_path: "./output/knowledgebase.json"
    # Hub alternatives:
    # train_dataset_id: "AdamLucek/quickb-qa"
    # knowledgebase_dataset_id: "AdamLucek/quickb-kb"

  # Training hyperparameters
  training_arguments:
    output_path: "./output/modernbert_quickb"
    device: "cuda" # Options: "cuda", "mps", "cpu"
    epochs: 4
    batch_size: 32
    gradient_accumulation_steps: 16
    learning_rate: 2.0e-5
    warmup_ratio: 0.1
    lr_scheduler_type: "cosine"
    optim: "adamw_torch_fused"
    tf32: true
    bf16: true
    batch_sampler: "no_duplicates"  # Options: "batch_sampler", "no_duplicates", "group_by_label"
    eval_strategy: "epoch"
    save_strategy: "epoch"
    logging_steps: 10
    save_total_limit: 3
    load_best_model_at_end: true
    report_to: "none"

  # Optional: Push trained model to Hub
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_model_id: "AdamLucek/modernbert-embed-quickb"
```

### Alternative Chunker Configurations

1. **Fixed Token Chunker**
```yaml
chunker_config:
  output_path: "./output/fixed_token_chunks.json"
  chunker: "FixedTokenChunker"
  chunker_arguments:
    encoding_name: "cl100k_base"
    chunk_size: 400
    chunk_overlap: 50
    length_type: "token"
```

2. **Cluster Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/semantic_clusters.json"
  chunker: "ClusterSemanticChunker"
  chunker_arguments:
    max_chunk_size: 400    # Max tokens after clustering
    min_chunk_size: 50     # Initial split size
    length_type: "token"
    litellm_config:
      embedding_model: "text-embedding-3-large"  # Required for semantic clustering
```

3. **LLM Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/llm_semantic_chunks.json"
  chunker: "LLMSemanticChunker"
  chunker_arguments:
    length_type: "token"
    litellm_config:
      model: "openai/gpt-4o"  # LLM for split decisions
```

4. **Kamradt Modified Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/kamradt_chunks.json"
  chunker: "KamradtModifiedChunker"
  chunker_arguments:
    avg_chunk_size: 400    # Target average size
    min_chunk_size: 50     # Minimum initial split
    length_type: "token"
    litellm_config: 
      embedding_model: "text-embedding-3-large"  # For similarity calculations
```
### Device Support

QuicKB supports multiple compute devices for model training:

- **CUDA**: NVIDIA GPUs (default and recommended for best performance)
- **MPS**: Apple Silicon on macOS
- **CPU**: Available on all systems, but significantly slower for training

To specify your preferred device, use the `device` parameter in your training configuration:

```yaml
training:
  training_arguments:
    device: "cuda"  # Options: "cuda", "mps", "cpu"
```

**Note**: The default configuration and hyperparameters are optimized for CUDA GPUs. When using MPS or CPU, you may need to install other torch versions or adjust the following parameters for optimal performance:

- Reduce `batch_size` (e.g., 8-16 for MPS, 4-8 for CPU)
- Reduce `gradient_accumulation_steps` for CPU training
- Set `bf16: false` and `tf32: false` for CPU training
- Use other optimizers like `adamw_torch`
- Consider using smaller base models

Device selection is automatic if you don't specify a device (prioritizing CUDA > MPS > CPU), but explicit configuration is recommended for reproducibility.

### LiteLLM Integration

QuicKB uses [LiteLLM](https://docs.litellm.ai/docs/) for flexible model provider integration, allowing you to use any supported LLM or embedding provider for question generation and semantic chunking. This enables both cloud-based and local model deployment.

The LiteLLM configuration is managed through the `litellm_config` section in both the chunking and question generation configurations:

```yaml
litellm_config:
  model: "openai/gpt-4o"                     # LLM model identifier
  model_api_base: null                       # Optional API base URL for LLM
  embedding_model: "text-embedding-3-large"  # Embedding model identifier
  embedding_api_base: null                   # Optional API base URL for embeddings
```

**Using Local Models**:

1. Set up an OpenAI API compatible endpoint (e.g., Ollama, vLLM)
2. Configure the `model_api_base` or `embedding_api_base` in your config
3. Use the appropriate model identifier format

Example local setup:

```yaml
# For question generation
question_generation:
  litellm_config:
    model: "local/llama-7b"
    model_api_base: "http://localhost:8000"
    embedding_model: "local/bge-small"
    embedding_api_base: "http://localhost:8000"

# For semantic chunkers
chunker_config:
  chunker: "ClusterSemanticChunker"  # or other semantic chunkers
  chunker_arguments:
    litellm_config:
      model: "local/llama-7b"
      model_api_base: "http://localhost:8000"
      embedding_model: "local/bge-small"
      embedding_api_base: "http://localhost:8000"
```

For more details on setting up local models and supported providers, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

### Hugging Face Hub Integration

QuicKB integrates directly with [Hugging Face](https://huggingface.co/) for storing and loading datasets and models. Each pipeline stage can optionally push its outputs to the Hub, and subsequent stages can load data directly from there.

The Hub integration is configured through `upload_config` sections and dataset source settings:

```yaml
# Example Hub configuration for chunking
chunker_config:
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "username/quickb-kb"

# Loading data from Hub for question generation
question_generation:
  input_dataset_config:
    dataset_source: "hub"
    knowledgebase_dataset_id: "username/quickb-kb"
  
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "username/quickb-qa"

# Loading from Hub for training
training:
  train_dataset_config:
    dataset_source: "hub"
    train_dataset_id: "username/quickb-qa"
    knowledgebase_dataset_id: "username/quickb-kb"
  
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_model_id: "username/modernbert-embed-quickb"
```
**Authentication**
- Set your Hugging Face token using the `HF_TOKEN` environment variable or specify it in the config using `hub_token`

## Output Format

### Knowledgebase Dataset
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": "Section 12.1: Termination clauses...",
  "source": "docs/contracts/2024/Q1-agreement.txt"
}
```

### Training Dataset
```json
{
  "anchor": "What are the termination notice requirements?",
  "positive": "Section 12.1: Either party may terminate...",
  "question_id": "a3b8c7d0-e83a-4b5c-b12d-3f7a8d4c9e1b",
  "chunk_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

## Environment Variables

- `<PROVIDER>_API_KEY`: Required for LLM embeddings, question generation, and chunking
- `HF_TOKEN`: Required for Hugging Face Hub uploads and downloads

## Citations

QuicKB builds upon these foundational works:

ChromaDB: [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
```bibtex
@techreport{smith2024evaluating,
  title = {Evaluating Chunking Strategies for Retrieval},
  author = {Smith, Brandon and Troynikov, Anton},
  year = {2024},
  month = {July},
  institution = {Chroma},
  url = {https://research.trychroma.com/evaluating-chunking},
}
```

Sentence Transformers
```bibtext
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

Todo List:

- Cleaner handling of config arguments and validation at pipeline stages
- pydantic v2 fields warning
- Custom Model Card (Using base from SBERT currently)

## License

MIT License - See [LICENSE](LICENSE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
