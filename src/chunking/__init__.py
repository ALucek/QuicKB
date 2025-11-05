from importlib import import_module
from typing import Any

from .registry import ChunkerRegistry, lazy_chunker_modules
from .utils import get_length_function, get_token_count, get_character_count

__all__ = [
    "ClusterSemanticChunker",
    "LLMSemanticChunker",
    "FixedTokenChunker",
    "RecursiveTokenChunker",
    "KamradtModifiedChunker",
    "ChunkerRegistry",
    "get_length_function",
    "get_token_count",
    "get_character_count",
]


def __getattr__(name: str) -> Any:
    if name in lazy_chunker_modules:
        module = import_module(lazy_chunker_modules[name])
        return getattr(module, name)
    raise AttributeError(f"module 'chunking' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)