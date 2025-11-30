"""
Logging and Telemetry module for GenAI Intelligence Studio.

Provides structured logging for:
- LLM calls
- ReAct agent steps
- Tool invocations
- Retrieval operations
- Mode detection
- Errors and exceptions
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import threading


# Create logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{color}[{timestamp}] [{record.levelname}] {record.name}: {record.getMessage()}{self.RESET}"


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler (JSON format)
    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings+ to console
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)
    
    return logger


# Create specialized loggers
react_logger = setup_logger("react_agent", "react.log")
retriever_logger = setup_logger("retriever", "retriever.log")
llm_logger = setup_logger("llm", "llm.log")
tool_logger = setup_logger("tools", "tools.log")
error_logger = setup_logger("errors", "errors.log", logging.ERROR)
mode_logger = setup_logger("mode", "mode.log")


class TelemetryTracker:
    """Track telemetry metrics for the application."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_metrics()
        return cls._instance
    
    def _init_metrics(self):
        self.metrics = {
            "llm_calls": 0,
            "llm_tokens_used": 0,
            "tool_calls": 0,
            "retrieval_calls": 0,
            "react_steps": 0,
            "errors": 0,
            "mode_counts": {"docs": 0, "video": 0, "product": 0},
            "avg_response_time": 0.0,
            "total_requests": 0,
        }
        self._response_times = []
    
    def track_llm_call(self, model: str, tokens: int = 0):
        self.metrics["llm_calls"] += 1
        self.metrics["llm_tokens_used"] += tokens
        llm_logger.info(f"LLM call: model={model}, tokens={tokens}")
    
    def track_tool_call(self, tool_name: str, query: str, success: bool = True):
        self.metrics["tool_calls"] += 1
        tool_logger.info(f"Tool call: {tool_name}, query={query[:100]}, success={success}")
    
    def track_retrieval(self, query: str, num_docs: int, scores: list = None):
        self.metrics["retrieval_calls"] += 1
        retriever_logger.info(f"Retrieval: query={query[:100]}, docs={num_docs}, scores={scores}")
    
    def track_react_step(self, step: int, action: str, observation: str = ""):
        self.metrics["react_steps"] += 1
        react_logger.info(f"ReAct step {step}: action={action}, observation={observation[:200]}")
    
    def track_mode(self, mode: str):
        self.metrics["mode_counts"][mode] = self.metrics["mode_counts"].get(mode, 0) + 1
        mode_logger.info(f"Mode detected: {mode}")
    
    def track_error(self, error: Exception, context: str = ""):
        self.metrics["errors"] += 1
        error_logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def track_response_time(self, seconds: float):
        self._response_times.append(seconds)
        self.metrics["total_requests"] += 1
        self.metrics["avg_response_time"] = sum(self._response_times) / len(self._response_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self._init_metrics()


# Global telemetry instance
telemetry = TelemetryTracker()


def log_llm_call(func):
    """Decorator to log LLM calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            telemetry.track_llm_call(model="groq", tokens=0)
            llm_logger.info(f"LLM call completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            telemetry.track_error(e, "llm_call")
            raise
    return wrapper


def log_tool_call(tool_name: str):
    """Decorator factory to log tool calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            query = args[0] if args else kwargs.get("query", "")
            try:
                result = func(*args, **kwargs)
                telemetry.track_tool_call(tool_name, str(query), success=True)
                return result
            except Exception as e:
                telemetry.track_tool_call(tool_name, str(query), success=False)
                telemetry.track_error(e, f"tool_{tool_name}")
                raise
        return wrapper
    return decorator


def log_retrieval(func):
    """Decorator to log retrieval operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get("query", "")
        try:
            result = func(*args, **kwargs)
            num_docs = len(result) if hasattr(result, "__len__") else 0
            telemetry.track_retrieval(str(query), num_docs)
            return result
        except Exception as e:
            telemetry.track_error(e, "retrieval")
            raise
    return wrapper


def log_react_step(step: int, action: str, observation: str = ""):
    """Log a ReAct agent step."""
    telemetry.track_react_step(step, action, observation)


def log_mode_detection(mode: str, question: str):
    """Log mode detection."""
    telemetry.track_mode(mode)
    mode_logger.info(f"Mode={mode}, Question={question[:100]}")


def get_log_summary() -> Dict[str, Any]:
    """Get a summary of all logs and metrics."""
    return {
        "metrics": telemetry.get_metrics(),
        "log_files": {
            "react": str(LOGS_DIR / "react.log"),
            "retriever": str(LOGS_DIR / "retriever.log"),
            "llm": str(LOGS_DIR / "llm.log"),
            "tools": str(LOGS_DIR / "tools.log"),
            "errors": str(LOGS_DIR / "errors.log"),
            "mode": str(LOGS_DIR / "mode.log"),
        }
    }
