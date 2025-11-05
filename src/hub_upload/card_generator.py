from typing import Any, Dict, Optional

class DatasetCardGenerator:
    """Handles dataset card generation for quickb datasets."""
    
    def __init__(self, template_path: str = "src/hub_upload/template.md"):
        """Initialize with path to card template."""
        self.template_path = template_path
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
    def _get_size_category(self, num_entries: Optional[int]) -> str:
        """Determine the size category based on number of entries."""
        if not num_entries:
            return "unknown"
        elif num_entries < 1000:
            return "n<1K"
        elif num_entries < 10000:
            return "1K<n<10K"
        elif num_entries < 100000:
            return "10K<n<100K"
        elif num_entries < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"
            
    def _format_chunker_params(self, params: Dict[str, Any]) -> str:
        """Simple Markdown-safe parameter formatting"""
        return "\n  ".join(
            f"- **{key}**: `{repr(value)}`" 
            for key, value in params.items() 
            if value is not None and not key.startswith('_')
        )
        
    def _format_question_generation(self, 
        model_name: str,
        similarity_threshold: float,
        num_questions: int,
        num_deduped: int
    ) -> str:
        """Format question generation section if enabled."""
        return f"""### Question Generation
- **Model**: {model_name}
- **Deduplication threshold**: {similarity_threshold}
- **Results**:
  - Total questions generated: {num_questions}
  - Questions after deduplication: {num_deduped}"""

    def _format_dataset_structure(self, has_chunks: bool, has_questions: bool) -> str:
        """Format dataset structure section based on available configurations."""
        if has_questions:
            return """### Dataset Structure
- `anchor`: The generated question
- `positive`: The text chunk containing the answer
- `question_id`: Unique identifier for the question
- `chunk_id`: Reference to the source chunk"""
        else:
            return """### Dataset Structure
This dataset contains the following fields:

- `text`: The content of each text chunk
- `source`: The source file path for the chunk
- `id`: Unique identifier for each chunk"""

    def _format_chunking_section(self,
        chunker_name: str,
        chunker_params: Dict[str, Any],
        num_chunks: int,
        avg_chunk_size: float,
        num_files: int
    ) -> str:
        """Format chunking section if enabled."""
        return f"""### Chunking Configuration
- **Chunker**: {chunker_name}
- **Parameters**:
  {self._format_chunker_params(chunker_params)}

### Dataset Statistics
- Total chunks: {num_chunks:,}
- Average chunk size: {avg_chunk_size:.1f} words
- Source files: {num_files}"""
        
    def generate_card(self,
        dataset_name: str,
        chunker_name: Optional[str] = None,
        chunker_params: Optional[Dict[str, Any]] = None,
        num_chunks: Optional[int] = None,
        avg_chunk_size: Optional[float] = None,
        num_files: Optional[int] = None,
        question_generation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a dataset card with the provided information."""
        
        # Load knowledgebase data to determine size category
        size_category = self._get_size_category(num_chunks)
        
        # Determine components and tags
        has_chunks = all(x is not None for x in [chunker_name, chunker_params, num_chunks, avg_chunk_size, num_files])
        has_questions = question_generation is not None
        
        gen_tag = "\n- question-generation" if has_questions else ""
        
        # Format sections based on available data
        chunking_section = ""
        if has_chunks:
            chunking_section = self._format_chunking_section(
                chunker_name=chunker_name,
                chunker_params=chunker_params,
                num_chunks=num_chunks,
                avg_chunk_size=avg_chunk_size,
                num_files=num_files
            )
            
        qg_section = ""
        if has_questions:
            qg_section = self._format_question_generation(
                model_name=question_generation["model_name"],
                similarity_threshold=question_generation["similarity_threshold"],
                num_questions=question_generation["num_questions"],
                num_deduped=question_generation["num_deduped"]
            )
            
        # Generate dataset structure section
        dataset_structure = self._format_dataset_structure(has_chunks, has_questions)
            
        # Fill template
        return self.template.format(
            dataset_name=dataset_name,
            gen_tag=gen_tag,
            size_category=size_category,
            chunker_section=chunking_section,
            question_generation=qg_section,
            dataset_structure=dataset_structure
        )