"""
Rate limiting utilities for API calls
"""

import asyncio
import time
from collections import deque
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter that enforces both requests-per-minute and concurrent request limits
    """
    
    def __init__(self, max_requests_per_minute: int = 60, max_concurrent_requests: int = 10):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        
        # Track request timestamps for rate limiting
        self.request_timestamps = deque()
        
        # Semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        logger.info(f"🚦 Rate limiter initialized: {max_requests_per_minute} req/min, {max_concurrent_requests} concurrent")
    
    async def acquire(self):
        """
        Acquire permission to make a request, enforcing both rate and concurrency limits
        """
        # For Claude, allow immediate start but limit at API level
        # This enables true concurrent request initiation
        return RateLimitContext(self)
    
    async def _enforce_rate_limit(self):
        """
        Enforce requests-per-minute rate limiting
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Calculate time to wait until oldest request expires
            oldest_request = self.request_timestamps[0]
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                logger.warning(f"🚦 Rate limit reached ({len(self.request_timestamps)}/{self.max_requests_per_minute}). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                # Recursively try again after waiting
                await self._enforce_rate_limit()
                return
        
        # Record this request timestamp
        self.request_timestamps.append(current_time)
        logger.debug(f"🚦 Rate limit check passed ({len(self.request_timestamps)}/{self.max_requests_per_minute} in last minute)")


class RateLimitContext:
    """
    Context manager for rate-limited operations
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.acquired = False
    
    async def __aenter__(self):
        # Acquire semaphore only when entering the context
        await self.rate_limiter.semaphore.acquire()
        self.acquired = True
        
        # Apply rate limiting when actually making the call
        await self.rate_limiter._enforce_rate_limit()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Release the semaphore when done
        if self.acquired:
            self.rate_limiter.semaphore.release()


class APIRateLimitManager:
    """
    Manages rate limiters for different API providers
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create separate rate limiters for each provider to avoid interference
        base_rate = config.api.max_requests_per_minute
        base_concurrent = config.api.max_concurrent_requests
        
        # Enterprise Claude account (3.5M tokens/min) - can handle very high concurrency
        # Running 3 Claude models simultaneously (claude-sonnet-4, claude-opus-4, claude-sonnet-3.7)
        openai_rate = max(1, base_rate // 3)  # Distribute evenly among 3 providers
        google_rate = max(1, base_rate // 3)  
        claude_rate = max(1, base_rate // 3)   # Claude gets equal share (enterprise account)
        
        # For 3 simultaneous Claude models with 200 concurrent each = 600 total concurrent needed
        openai_concurrent = max(1, base_concurrent // 3)
        google_concurrent = max(1, base_concurrent // 3) 
        claude_concurrent = max(600, base_concurrent * 2)  # Claude gets at least 600 concurrent for 3 models
        
        self.limiters = {
            "openai": RateLimiter(openai_rate, openai_concurrent),
            "google": RateLimiter(google_rate, google_concurrent),
            "claude": RateLimiter(claude_rate, claude_concurrent)
        }
        
        logger.info(f"📊 API rate limiters created - OpenAI: {openai_rate}/{openai_concurrent}, Gemini: {google_rate}/{google_concurrent}, Claude: {claude_rate}/{claude_concurrent} (req/min, concurrent)")
    
    async def acquire(self, provider: str):
        """
        Acquire rate limit permission for a specific provider
        """
        if provider not in self.limiters:
            # Default limiter for unknown providers
            provider = "openai"
        
        return await self.limiters[provider].acquire() 