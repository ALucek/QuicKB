
# This script is adapted from the Greg Kamradt's notebook on chunking.
# Original code can be found at: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
# chunking_evaluation modification: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/kamradt_modified_chunker.py

from typing import List, Optional

import numpy as np
from litellm import embedding

from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
from .registry import ChunkerRegistry

@ChunkerRegistry.register("KamradtModifiedChunker")
class KamradtModifiedChunker(BaseChunker):
    def __init__(
        self,
        avg_chunk_size: int = 400,
        min_chunk_size: int = 50,
        litellm_config: Optional[dict] = None,
        length_type: str = 'token',
        **kwargs
    ):
        super().__init__(length_type=length_type, **kwargs)

        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function
        )

        self._litellm_config = litellm_config or {}
        self.avg_chunk_size = avg_chunk_size

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using LiteLLM."""
        response = embedding(
            model=self._litellm_config.get('embedding_model', 'text-embedding-3-large'),
            input=texts,
            api_base=self._litellm_config.get('embedding_api_base')
        )
        # Extract embeddings from the response data
        if hasattr(response, 'data'):
            return [item['embedding'] for item in response.data]
        elif isinstance(response, dict) and 'data' in response:
            return [item['embedding'] for item in response['data']]
        raise ValueError(f"Unexpected response format from LiteLLM: {response}")

    def combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        for i in range(len(sentences)):
            combined = []
            for j in range(max(0, i - buffer_size), min(len(sentences), i + buffer_size + 1)):
                combined.append(sentences[j]['sentence'])
            sentences[i]['combined_sentence'] = ' '.join(combined)
        return sentences

    def calculate_cosine_distances(self, sentences: List[dict]):
        embeddings = []
        for i in range(0, len(sentences), 500):
            batch = [s['combined_sentence'] for s in sentences[i:i + 500]]
            embeddings.extend(self._get_embeddings(batch))

        embedding_matrix = np.array(embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix /= norms
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - similarity_matrix[i, i + 1]
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def split_text(self, text: str) -> List[str]:
        s_list = self.splitter.split_text(text)
        sentences = [{'sentence': s, 'index': i} for i, s in enumerate(s_list)]
        if not sentences:
            return []

        sentences = self.combine_sentences(sentences, 3)
        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(self.length_function(s['sentence']) for s in sentences)
        target_splits = total_tokens // self.avg_chunk_size if self.avg_chunk_size else 1
        distances = np.array(distances)

        low, high = 0.0, 1.0
        while high - low > 1e-6:
            mid = (low + high) / 2
            if (distances > mid).sum() > target_splits:
                low = mid
            else:
                high = mid

        split_indices = [i for i, d in enumerate(distances) if d > high]
        chunks = []
        start = 0

        for idx in split_indices:
            chunks.append(' '.join(s['sentence'] for s in sentences[start:idx + 1]))
            start = idx + 1

        if start < len(sentences):
            chunks.append(' '.join(s['sentence'] for s in sentences[start:]))

        return chunks