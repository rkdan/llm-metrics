import functools
from datetime import datetime
from typing import Any, Callable

import numpy as np

from .base import Loggable, Metrics


class Perplexity(Loggable):
    """Collection of metric decorators for model response analysis."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs
    ):
        """Initialize the Metrics class.

        Args:
            log (bool, optional): Whether to log metrics. Defaults to True.
            output_dir (str, optional): Directory for log files. Defaults to "./logs".
            experiment_name (str, optional): Name of experiment. Defaults to "experiment".
        """
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator to calculate and log perplexity of model responses.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The wrapped function that includes perplexity calculation.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:

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
                            metric_category="generative",
                        )
                    except AttributeError:
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
                            metric_category="generative",
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
                        metric_category="generative",
                    )
                raise

        return wrapper


class GenerativeMetrics(Metrics):
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
        self._perplexity = Perplexity(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )

        # Mapping of metric names to their implementations
        self._metrics = {
            "perplexity": lambda: self._perplexity,
        }
