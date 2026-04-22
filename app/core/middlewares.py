"""
Middleware configurations and implementations for the AI Proctoring application.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict, Tuple
from collections import defaultdict

from dotenv import load_dotenv

from app.core.config import RATE_LIMIT_CONFIG, CORS_CONFIG

load_dotenv()

# In-memory storage for rate limiting
# In production, consider using Redis or another persistent store
rate_limit_storage: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))

class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, requests_per_minute: int = 60, window_size: int = 60):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        client_ip = self._get_client_ip(request)
        
        if not self._is_rate_limited(client_ip):
            await self.app(scope, receive, send)
        else:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Maximum 60 requests per minute allowed.",
                    "retry_after": self._get_retry_after(client_ip)
                }
            )
            await response(scope, receive, send)
    
    def _get_client_ip(self, request: Request) -> str:
        _=self
        """Extract client IP address from request"""
        # Check for forwarded IP first (in case of proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited"""
        current_time = time.time()
        request_count, window_start = rate_limit_storage[client_ip]
        
        # Reset window if it has expired
        if current_time - window_start >= self.window_size:
            rate_limit_storage[client_ip] = (1, current_time)
            return False
        
        # Check if limit exceeded
        if request_count >= self.requests_per_minute:
            return True
        
        # Increment request count
        rate_limit_storage[client_ip] = (request_count + 1, window_start)
        return False
    
    def _get_retry_after(self, client_ip: str) -> int:
        """Get seconds until rate limit resets"""
        _, window_start = rate_limit_storage[client_ip]
        return int(self.window_size - (time.time() - window_start))

def setup_rate_limit_middleware(app: FastAPI) -> None:
    """Setup rate limiting middleware for the FastAPI application"""
    if RATE_LIMIT_CONFIG.ENABLED:
        app.add_middleware(
            RateLimitMiddleware, # type: ignore
            requests_per_minute=RATE_LIMIT_CONFIG.REQUESTS_PER_MINUTE,
            window_size=RATE_LIMIT_CONFIG.WINDOW_SIZE_SECONDS
        )

def setup_cors_middleware(app: FastAPI) -> None:
    """Setup CORS middleware for the FastAPI application"""
    app.add_middleware(
        CORSMiddleware, # type: ignore
        allow_origins=CORS_CONFIG.ALLOW_ORIGINS,
        allow_credentials=CORS_CONFIG.ALLOW_CREDENTIALS,
        allow_methods=CORS_CONFIG.ALLOW_METHODS,
        allow_headers=CORS_CONFIG.ALLOW_HEADERS,

    )
