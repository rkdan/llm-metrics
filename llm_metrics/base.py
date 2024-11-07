import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .utils import combine_metrics


class Loggable:
    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        self.log = log

        if self.log:
            self.session_start = datetime.now()
            self.experiment_name = experiment_name

            date_str = self.session_start.strftime("%Y-%m-%d")
            self.session_dir = Path(output_dir) / self.experiment_name / date_str
            # Create the base session directory
            self.session_dir.mkdir(parents=True, exist_ok=True)

    def _write_log(
        self, log_type: str, data: dict, metric_category: Optional[str] = None
    ):
        """Write a log entry to a JSONL file.

        Args:
            log_type (str): Type of log entry (e.g. "latency", "bleu", "perplexity")
            data (dict): Log entry data
            metric_category (str, optional): Metric category
                ["system", "generative", "similarity"]. Defaults to None.
        """
        if metric_category:
            # For metrics, use metrics/category/type.jsonl
            file_path = (
                self.session_dir / "metrics" / metric_category / f"{log_type}.jsonl"
            )
        else:
            # For API logs, use api/type.jsonl
            file_path = self.session_dir / "api" / f"{log_type}.jsonl"

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "a") as f:
            json.dump(data, f)
            f.write("\n")

    def log_off(self):
        """_summary_"""
        self.log = False

    def log_on(self):
        """_summary_"""
        self.log = True


class Metrics:
    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        self.settings = {
            "log": log,
            "output_dir": output_dir,
            "experiment_name": experiment_name,
        }

        self._metrics = {}  # type: dict[str, Callable]

    def __call__(self, metrics: str | list[str]) -> Callable:
        """Enable metrics for a function.

        Args:
            metrics (str | list[str]): List of metrics to enable.

        Raises:
            ValueError: At least one metric must be specified
            ValueError: Unknown metric entered

        Returns:
            Callable: Decorated function
        """
        if isinstance(metrics, str):
            metrics = [metrics]

        if not metrics:
            raise ValueError("At least one metric must be specified")

        metric_decorators = []
        for metric in metrics:
            if metric not in self._metrics:
                raise ValueError(
                    f"Unknown metric: {metric}. Available metrics: {', '.join(self._metrics.keys())}"
                )
            metric_decorators.append(self._metrics[metric]())

        return combine_metrics([m.__call__ for m in metric_decorators])
