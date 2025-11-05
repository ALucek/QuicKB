from abc import ABC, abstractmethod
from typing import List

from .utils import get_length_function

class BaseChunker(ABC):
    """Base class for all chunking implementations."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the chunker with length function configuration.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self._initialize_length_function(kwargs)

    def _initialize_length_function(self, kwargs):
        """
        Initialize the length function based on provided configuration.
        
        Args:
            kwargs: Keyword arguments that may contain length function configuration
        """
        # Get length function type and parameters
        length_type = kwargs.pop('length_type', 'token')
        encoding_name = kwargs.pop('encoding_name', 'cl100k_base')
        
        # Set up default length function
        self.length_function = get_length_function(
            length_type=length_type,
            encoding_name=encoding_name
        )
        
        # Override with custom length function if provided
        if 'length_function' in kwargs:
            self.length_function = kwargs.pop('length_function')

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split input text into chunks.
        
        Args:
            text: The input text to split
            
        Returns:
            List of text chunks
        """
        pass