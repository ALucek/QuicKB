import json
import logging
import os
from typing import Any, Dict, Optional
from datasets import Dataset
from huggingface_hub import create_repo, upload_file, repo_exists
from .card_generator import DatasetCardGenerator

logger = logging.getLogger(__name__)

class DatasetPusher:
    """Handles uploading datasets to the Hugging Face Hub."""

    def __init__(self, username: str, token: Optional[str] = None):
        """Initialize with HF credentials."""
        self.username = username
        self.token = token or os.getenv("HF_TOKEN")

        if not self.token:
            raise ValueError("No Hugging Face token provided or found in environment")

        self.card_generator = DatasetCardGenerator()

    def _repository_exists(self, repo_id: str) -> bool:
        """Check if repository already exists."""
        try:
            return repo_exists(repo_id, repo_type="dataset")
        except Exception as e:
            logger.error(f"Error checking repository: {str(e)}")
            return False

    def _load_json_file(self, file_path: str) -> list:
        """Load JSON file and ensure it's a list of records."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def _calculate_dataset_stats(self, knowledgebase_data: list) -> Dict[str, Any]:
        """Calculate statistics for dataset card."""
        text_lengths = [len(item['text'].split()) for item in knowledgebase_data]
        unique_sources = len(set(item['source'] for item in knowledgebase_data))

        return {
            'num_chunks': len(knowledgebase_data),
            'avg_chunk_size': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'num_files': unique_sources
        }

    def push_dataset(
        self,
        hub_dataset_id: str,
        knowledgebase_path: Optional[str] = None,
        chunker_info: Optional[Dict[str, Any]] = None,
        train_path: Optional[str] = None,
        question_gen_info: Optional[Dict[str, Any]] = None,
        private: bool = True
    ) -> None:
        """Push dataset to the Hugging Face Hub, overwriting existing data."""
        try:
            # Create repository if it doesn't exist
            if not self._repository_exists(hub_dataset_id):
                create_repo(
                    hub_dataset_id,
                    repo_type="dataset",
                    private=private,
                    token=self.token
                )
                logger.info(f"Created new dataset repository: {hub_dataset_id}")
            else:
                logger.info(f"Dataset repository exists: {hub_dataset_id}")

            # Load and push knowledgebase if provided
            kb_data = None
            if knowledgebase_path:
                kb_data = self._load_json_file(knowledgebase_path)
                kb_dataset = Dataset.from_list(kb_data)
                kb_dataset.push_to_hub(
                    hub_dataset_id,
                    token=self.token,
                    private=private
                )
                logger.info(f"Pushed knowledgebase to {hub_dataset_id}")

            # Load and push training data if provided
            if train_path:
                train_data = self._load_json_file(train_path)
                train_dataset = Dataset.from_list(train_data)
                train_dataset.push_to_hub(
                    hub_dataset_id,
                    token=self.token,
                    private=private
                )
                logger.info(f"Pushed training data to {hub_dataset_id}")

            # Generate and upload README
            repository_name = hub_dataset_id.split('/')[-1]
            card_content = self.card_generator.generate_card(
                dataset_name=repository_name,
                chunker_name=chunker_info.get('chunker_name') if chunker_info else None,
                chunker_params=chunker_info.get('chunker_params') if chunker_info else None,
                num_chunks=self._calculate_dataset_stats(kb_data)['num_chunks'] if kb_data else None,
                avg_chunk_size=self._calculate_dataset_stats(kb_data)['avg_chunk_size'] if kb_data else None,
                num_files=self._calculate_dataset_stats(kb_data)['num_files'] if kb_data else None,
                question_generation=question_gen_info
            )

            upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=hub_dataset_id,
                repo_type="dataset",
                token=self.token
            )
            logger.info(f"Uploaded README.md to {hub_dataset_id}")

        except Exception as e:
            logger.error(f"Error pushing dataset to Hub: {str(e)}")
            raise