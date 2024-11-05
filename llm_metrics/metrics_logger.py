import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


class Metrics:
    """Collection of metric decorators for model response analysis."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
    ):
        """
        Initialize the metrics collection.

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

    def _write_log(self, metric_name: str, data: dict):
        """Write a metric log entry to the appropriate file."""
        file_path = self.session_dir / f"metric_{metric_name}.jsonl"
        with open(file_path, "a") as f:
            json.dump(data, f)
            f.write("\n")

    def perplexity(self, func: Callable) -> Callable:
        """
        Decorator that calculates and logs perplexity of model responses.

        Args:
            func: The function to be wrapped

        Returns:
            Wrapped function that logs perplexity
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Call the original function
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        # Try to calculate perplexity if logprobs are available
                        logprobs = [
                            token.logprob
                            for token in result.choices[0].logprobs.content
                        ]

                        if logprobs is not None:
                            perplexity = np.exp(-np.mean(logprobs))
                        else:
                            perplexity = None

                        self._write_log(
                            "perplexity",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "perplexity": perplexity,
                                "success": True,
                            },
                        )
                    except AttributeError:
                        # Log if logprobs weren't available
                        self._write_log(
                            "perplexity",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "perplexity": None,
                                "success": True,
                                "note": "logprobs not available in response",
                            },
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "perplexity",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                    )
                raise

        return wrapper
