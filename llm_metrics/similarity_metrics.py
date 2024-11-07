# similarity_metrics.py
import functools
from datetime import datetime
from typing import Any, Callable

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .base import Loggable, Metrics


class RougeMetric(Loggable):
    """ROUGE metric implementation."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def _get_lcs_length(self, x: list[str], y: list[str]) -> np.ndarray:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = np.zeros((m + 1, n + 1))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def __call__(self, func: Callable) -> Callable:
        """Decorator to calculate and log ROUGE-L scores."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if "reference_text" not in kwargs:
                raise ValueError(
                    "reference_text must be provided as a keyword argument"
                )

            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        generated_text = result.choices[0].message.content
                        reference_text = kwargs.get("reference_text")

                        if generated_text and reference_text:
                            # Tokenize into words
                            hyp_words = generated_text.lower().split()
                            ref_words = reference_text.lower().split()

                            if not hyp_words or not ref_words:
                                scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                            else:
                                lcs_length = self._get_lcs_length(hyp_words, ref_words)

                                precision = (
                                    lcs_length / len(hyp_words) if hyp_words else 0.0
                                )
                                recall = (
                                    lcs_length / len(ref_words) if ref_words else 0.0
                                )

                                if precision + recall == 0:
                                    f1 = 0.0
                                else:
                                    f1 = 2 * precision * recall / (precision + recall)

                                scores = {
                                    "precision": float(precision),
                                    "recall": float(recall),
                                    "f1": float(f1),
                                }

                            self._write_log(
                                "rouge_l",
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "function": func.__name__,
                                    "module": func.__module__,
                                    **scores,
                                    "success": True,
                                },
                                metric_category="similarity",
                            )
                    except AttributeError as e:
                        self._write_log(
                            "rouge_l",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": str(e),
                            },
                            metric_category="similarity",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "rouge_l",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        metric_category="similarity",
                    )
                raise

        return wrapper


class BleuMetric(Loggable):
    """BLEU metric implementation."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

        self.sentence_bleu = sentence_bleu
        self.smoothing = SmoothingFunction().method1

    def __call__(self, func: Callable) -> Callable:
        """Decorator to calculate and log BLEU scores."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if "reference_text" not in kwargs:
                raise ValueError(
                    "reference_text must be provided as a keyword argument"
                )

            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        generated_text = result.choices[0].message.content
                        reference_text = kwargs.get("reference_text")

                        if generated_text and reference_text:
                            # Tokenize
                            hypothesis = generated_text.lower().split()
                            reference = [reference_text.lower().split()]

                            # Calculate BLEU score
                            bleu_score = self.sentence_bleu(
                                reference, hypothesis, smoothing_function=self.smoothing
                            )

                            self._write_log(
                                "bleu",
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "function": func.__name__,
                                    "module": func.__module__,
                                    "bleu_score": float(bleu_score),
                                    "success": True,
                                },
                                metric_category="similarity",
                            )
                    except AttributeError as e:
                        self._write_log(
                            "bleu",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": str(e),
                            },
                            metric_category="similarity",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "bleu",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        metric_category="similarity",
                    )
                raise

        return wrapper


class SimilarityMetrics(Metrics):
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

        self.settings = {
            "log": log,
            "output_dir": output_dir,
            "experiment_name": experiment_name,
        }

        # Initialize basic metrics
        self._rouge = RougeMetric(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )
        self._bleu = None  # Lazy loading

        # Mapping of metric names to their implementations
        self._metrics = {"rouge": lambda: self._rouge, "bleu": lambda: self.bleu}

    @property
    def bleu(self):
        """Lazy loading of BLEU metric."""
        if self._bleu is None:
            self._bleu = BleuMetric(**self.settings)
        return self._bleu
