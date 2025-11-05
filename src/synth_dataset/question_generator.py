from threading import Lock, RLock
from typing import Dict, List, Optional
import json
import uuid
import logging
import backoff
from litellm import completion, embedding
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .deduplicator import QuestionDeduplicator
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

class QuestionGenerator:
    def __init__(
        self,
        prompt_path: Optional[str] = None,
        prompt: Optional[str] = None,
        api_key: str = None,
        llm_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        dedup_enabled: bool = True,
        similarity_threshold: float = 0.85,
        max_workers: int = 20,
        model_api_base: str = None,
        embedding_api_base: str = None,
        embedding_batch_size: int = 500,
        llm_calls_per_minute: int = 15,
        embedding_calls_per_minute: int = 15,
        temperature: Optional[float] = None
    ):
        # Initialize basic attributes
        if prompt is not None and prompt_path is not None:
            raise ValueError("Provide either 'prompt' or 'prompt_path', not both.")

        if prompt is not None:
            self.prompt = prompt
        else:
            if prompt_path is None:
                raise ValueError("Either 'prompt' or 'prompt_path' must be provided.")
            self.prompt = self._load_prompt(prompt_path)

        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.dedup_enabled = dedup_enabled
        self.max_workers = max_workers
        self.model_api_base = model_api_base
        self.embedding_api_base = embedding_api_base
        self.embedding_batch_size = embedding_batch_size
        self.temperature = temperature

        # Thread safety mechanisms
        self._question_cache: Dict[str, List[Dict]] = {}
        self._cache_lock = RLock()  # Use RLock for recursive locking capability
        self._embedding_lock = Lock()  # Separate lock for embedding operations

        # Initialize deduplicator if enabled
        self.deduplicator = QuestionDeduplicator(similarity_threshold) if dedup_enabled else None

        # Initialize rate limiters with their own internal locks
        self.llm_rate_limiter = RateLimiter(llm_calls_per_minute, name="LLM") if llm_calls_per_minute is not None else None
        self.embedding_rate_limiter = RateLimiter(embedding_calls_per_minute, name="Embedding") if embedding_calls_per_minute is not None else None

    def _load_prompt(self, path: str) -> str:
        """Load the prompt template from a file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {path}")
            raise
        except IOError as e:
            logger.error(f"Error reading prompt template: {str(e)}")
            raise

    def _get_from_cache(self, chunk: str) -> Optional[List[Dict]]:
        """Thread-safe cache retrieval."""
        with self._cache_lock:
            return self._question_cache.get(chunk)

    def _add_to_cache(self, chunk: str, questions: List[Dict]) -> None:
        """Thread-safe cache addition."""
        with self._cache_lock:
            self._question_cache[chunk] = questions

    @backoff.on_exception(
        backoff.expo, 
        Exception, 
        max_tries=3,
        max_time=30  # Maximum total time to try
    )
    def _generate(self, chunk: str) -> str:
        """Generate questions for a single chunk with rate limiting."""
        # Wait for rate limit if needed
        if self.llm_rate_limiter:
            self.llm_rate_limiter.wait_if_needed()
        
        completion_kwargs = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Text: {chunk}"}
            ],
            "api_key": self.api_key,
            "timeout": 10  # Add timeout for API calls
        }
        
        if self.temperature is not None:
            completion_kwargs["temperature"] = self.temperature

        if self.model_api_base:
            completion_kwargs["api_base"] = self.model_api_base
            
        response = completion(**completion_kwargs)
        return response.choices[0].message.content

    def generate_for_chunk(self, chunk: str) -> List[Dict]:
        """Generate questions for a single chunk with caching."""
        # Check cache first
        cached_questions = self._get_from_cache(chunk)
        if cached_questions is not None:
            return cached_questions

        try:
            response = self._generate(chunk)
            questions = json.loads(response)["questions"]
            
            # Process questions
            processed_questions = []
            for q in questions:
                q.update({
                    "id": str(uuid.uuid4()),
                    "chunk_text": chunk,
                })
                q.pop("explanation", None)
                processed_questions.append(q)

            # Add to cache
            self._add_to_cache(chunk, processed_questions)
            return processed_questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _process_embeddings_batch(self, questions_batch: List[str]) -> List[List[float]]:
        """Process a batch of questions for embeddings with thread safety."""
        with self._embedding_lock:
            if self.embedding_rate_limiter:
                self.embedding_rate_limiter.wait_if_needed()
            
            embedding_kwargs = {
                "model": self.embedding_model,
                "input": questions_batch,
                "api_key": self.api_key,
                "timeout": 10
            }
            
            if self.embedding_api_base:
                embedding_kwargs["api_base"] = self.embedding_api_base
            
            response = embedding(**embedding_kwargs)
            return [data["embedding"] for data in response.data]

    def generate_for_chunks(self, chunks: List[str]) -> List[Dict]:
        """Generate questions for multiple chunks with thread safety."""
        results = []
        uncached_chunks = []

        # Check cache first
        for chunk in chunks:
            cached_results = self._get_from_cache(chunk)
            if cached_results is not None:
                results.extend(cached_results)
            else:
                uncached_chunks.append(chunk)

        if uncached_chunks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self.generate_for_chunk, chunk): chunk 
                    for chunk in uncached_chunks
                }
                
                with tqdm(total=len(uncached_chunks), desc="Generating questions") as pbar:
                    for future in as_completed(future_to_chunk):
                        try:
                            questions = future.result()
                            results.extend(questions)
                        except Exception as e:
                            chunk = future_to_chunk[future]
                            logger.error(f"Error processing chunk {chunk[:50]}...: {str(e)}")
                        pbar.update(1)
        
        if self.dedup_enabled and self.deduplicator and results:
            # Process embeddings in batches
            questions_text = [q["question"] for q in results]
            all_embeddings = []
            
            # Calculate total number of batches for the progress bar
            num_batches = (len(questions_text) + self.embedding_batch_size - 1) // self.embedding_batch_size
            
            # Add progress bar for embedding batches
            with tqdm(total=num_batches, desc="Processing embeddings", unit="batch") as pbar:
                for i in range(0, len(questions_text), self.embedding_batch_size):
                    batch = questions_text[i:i + self.embedding_batch_size]
                    try:
                        batch_embeddings = self._process_embeddings_batch(batch)
                        all_embeddings.extend(batch_embeddings)
                        pbar.update(1)
                        
                        # Optional: add batch statistics to progress bar
                        pbar.set_postfix({
                            'batch_size': len(batch),
                            'total_embedded': len(all_embeddings)
                        })
                        
                    except Exception as e:
                        logger.error(f"Error during embedding batch {i//self.embedding_batch_size}: {str(e)}")
                        return results  # Return un-deduplicated results on error
            
            # Deduplicate with thread safety
            with self._cache_lock:
                results = self.deduplicator.deduplicate(results, all_embeddings)
        
        return results