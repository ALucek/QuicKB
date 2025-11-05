
# This script is adapted from the chunking_evaluation package, developed by ChromaDB Research.
# Original code can be found at: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/llm_semantic_chunker.py
# License: MIT License

from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
import backoff
from tqdm import tqdm
from typing import List, Optional
import re
from .registry import ChunkerRegistry
from litellm import completion

@ChunkerRegistry.register("LLMSemanticChunker")
class LLMSemanticChunker(BaseChunker):
    def __init__(
        self, 
        litellm_config: Optional[dict] = None,
        length_type: str = 'token',
        **kwargs
    ):
        super().__init__(length_type=length_type, **kwargs)
        
        self._litellm_config = litellm_config or {}
        
        # Initialize the base splitter for initial text splitting
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=self.length_function
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        """Generate the prompt for the LLM."""
        base_prompt = (
            "You are an assistant specialized in splitting text into thematically consistent sections. "
            "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
            "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
            "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
            "Your response should be in the form: 'split_after: 3, 5'."
        )

        user_content = (
            f"CHUNKED_TEXT: {chunked_input}\n\n"
            f"Respond with split points (ascending, ≥{current_chunk}). "
            "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER"
        )
        if invalid_response:
            user_content += (
                f"\n\\Previous invalid response: {invalid_response}. "
                "DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again."
            )

        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_content}
        ]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _get_llm_response(self, context: str, current: int) -> str:
        """Get chunking suggestions from LLM using LiteLLM."""
        try:
            completion_kwargs = {
                "model": self._litellm_config.get('model', 'openai/gpt-4o'),
                "messages": self.get_prompt(context, current),
                "max_tokens": 200,
            }

            api_base = self._litellm_config.get('model_api_base')
            if api_base:
                completion_kwargs["api_base"] = api_base

            temperature = self._litellm_config.get('temperature')
            if temperature is not None:
                completion_kwargs["temperature"] = temperature

            response = completion(**completion_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return ""

    def _parse_response(self, response: str, current_chunk: int) -> List[int]:
        numbers = []
        if 'split_after:' in response:
            numbers = list(map(int, re.findall(r'\d+', response.split('split_after:')[1])))
        return sorted(n for n in numbers if n > current_chunk)  # Ensure 1-based > current 0-based

    def _merge_chunks(self, chunks: List[str], indices: List[int]) -> List[str]:
        """Merge chunks based on split indices (indices are 1-based from LLM)"""
        merged = []
        current = []
        # Convert to 0-based indices and sort
        split_points = sorted([i-1 for i in indices if i > 0])
        
        for i, chunk in enumerate(chunks):
            current.append(chunk)
            if i in split_points:
                merged.append(" ".join(current).strip())
                current = []
        if current:
            merged.append(" ".join(current).strip())
        return merged

    def split_text(self, text: str) -> List[str]:
        """Split input text into coherent chunks using LLM guidance."""
        chunks = self.splitter.split_text(text)
        split_indices = []
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while current_chunk < len(chunks) - 4:
                context_window = []
                token_count = 0

                for i in range(current_chunk, len(chunks)):
                    token_count += self.length_function(chunks[i])
                    if token_count > 800:
                        break
                    context_window.append(f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>")

                response = self._get_llm_response("\n".join(context_window), current_chunk)
                numbers = self._parse_response(response, current_chunk)  # FIXED: Added current_chunk argument

                if numbers:
                    split_indices.extend(numbers)
                    current_chunk = numbers[-1]
                    pbar.update(current_chunk - pbar.n)
                else:
                    break

        return self._merge_chunks(chunks, split_indices)