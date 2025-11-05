from threading import Lock
from time import sleep, time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter using a rolling window."""
    
    def __init__(self, calls_per_minute: int, name: str = "default"):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
            name: Name for this rate limiter instance (for logging)
        """
        self.calls_per_minute = calls_per_minute
        self.name = name
        self.window_size = 60  # 1 minute in seconds
        self.timestamps = deque(maxlen=calls_per_minute)
        self.lock = Lock()
    
    def _clean_old_timestamps(self, current_time: float) -> None:
        """Remove timestamps older than the window size."""
        cutoff_time = current_time - self.window_size
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
    
    def wait_if_needed(self) -> None:
        """
        Check if rate limit is reached and wait if necessary.
        Thread-safe implementation.
        """
        with self.lock:
            current_time = time()
            self._clean_old_timestamps(current_time)
            
            if len(self.timestamps) >= self.calls_per_minute:
                # Calculate required wait time
                oldest_timestamp = self.timestamps[0]
                wait_time = oldest_timestamp + self.window_size - current_time
                
                if wait_time > 0:
                    logger.debug(f"{self.name} rate limiter: Waiting {wait_time:.2f} seconds")
                    sleep(wait_time)
                    current_time = time()  # Update current time after sleep
            
            # Add new timestamp
            self.timestamps.append(current_time)
    
    def acquire(self) -> None:
        """Alias for wait_if_needed for more intuitive API."""
        self.wait_if_needed()
    
    def remaining_calls(self) -> int:
        """Return number of remaining calls allowed in current window."""
        with self.lock:
            self._clean_old_timestamps(time())
            return self.calls_per_minute - len(self.timestamps)