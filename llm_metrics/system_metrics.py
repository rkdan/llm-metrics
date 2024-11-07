import functools
import time
from datetime import datetime
from typing import Any, Callable

from .base import Loggable, Metrics


class Latency(Loggable):
    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs
    ):
        # Call parent class constructor
        super().__init__(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )

    def __call__(self, func: Callable) -> Callable:
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


class SystemMetrics(Metrics):
    """Factory class for similarity metrics with list-style decoration."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

        # Initialize basic metrics
        self._latency = Latency(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )

        self._metrics = {
            "latency": lambda: self._latency,
        }
