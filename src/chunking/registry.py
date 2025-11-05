from importlib import import_module
from typing import Callable, Dict, Type


# Map chunker names to their implementing modules. These are imported lazily to
# avoid triggering heavy optional dependencies (for example, LiteLLM pulling in
# Pydantic configs that emit warnings) when the registry is imported.
lazy_chunker_modules: Dict[str, str] = {
    "ClusterSemanticChunker": "chunking.cluster_semantic_chunker",
    "LLMSemanticChunker": "chunking.llm_semantic_chunker",
    "FixedTokenChunker": "chunking.fixed_token_chunker",
    "RecursiveTokenChunker": "chunking.recursive_token_chunker",
    "KamradtModifiedChunker": "chunking.kamradt_modified_chunker",
}


class ChunkerRegistry:
    _chunkers: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def decorator(chunker_class: Type) -> Type:
            cls._chunkers[name] = chunker_class
            return chunker_class

        return decorator

    @classmethod
    def _import_chunker(cls, name: str) -> None:
        module_path = lazy_chunker_modules.get(name)
        if module_path:
            import_module(module_path)

    @classmethod
    def _ensure_loaded(cls, name: str) -> None:
        if name in cls._chunkers:
            return
        cls._import_chunker(name)

    @classmethod
    def get_chunker(cls, name: str):
        cls._ensure_loaded(name)
        if name not in cls._chunkers:
            available = sorted(set(cls._chunkers.keys()) | set(lazy_chunker_modules.keys()))
            raise ValueError(f"Unknown chunker: {name}. Available chunkers: {available}")
        return cls._chunkers[name]

    @classmethod
    def available_chunkers(cls) -> list[str]:
        return sorted(set(cls._chunkers.keys()) | set(lazy_chunker_modules.keys()))