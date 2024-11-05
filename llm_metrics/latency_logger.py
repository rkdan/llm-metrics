import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class LatencyLogger:
    """A decorator class for logging function execution times."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
    ):
        """
        Initialize the latency logger.

        Args:
            log: Whether to enable logging (default: True)
            output_dir: Directory where logs will be stored (default: './logs')
            experiment_name: Optional name for this experiment
        """
        self.log = log

        if self.log:
            self.session_start = datetime.now()
            self.experiment_name = experiment_name

            date_str = self.session_start.strftime("%Y-%m-%d")
            self.session_dir = Path(output_dir) / self.experiment_name / date_str
            self.session_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator that wraps a function and logs its execution time.

        Args:
            func: The function to be wrapped

        Returns:
            Wrapped function that logs execution time
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if self.log:
                    self._write_log(
                        "latency",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "duration_seconds": duration,
                            "success": True,
                            "args_count": len(args),
                            "kwargs_count": len(kwargs),
                        },
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time

                if self.log:
                    self._write_log(
                        "latency",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "duration_seconds": duration,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "args_count": len(args),
                            "kwargs_count": len(kwargs),
                        },
                    )
                raise

        return wrapper

    def _write_log(self, log_type: str, data: dict):
        """Write a log entry to the appropriate file."""
        file_path = self.session_dir / f"{log_type}.jsonl"
        with open(file_path, "a") as f:
            json.dump(data, f)
            f.write("\n")
