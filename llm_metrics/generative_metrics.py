import functools
from collections import Counter
from datetime import datetime
from typing import Any, Callable

import nltk
import numpy as np

from .base import Loggable, Metrics

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


class Perplexity(Loggable):
    """Perplexity metric implementation."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        """Initialize the Perplexity metric.

        Args:
            log (bool, optional): To log or not to log? Defaults to True.
            output_dir (str, optional): Where to log. Defaults to "./logs".
            experiment_name (str, optional): What to log. Defaults to "experiment".
        """
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator to calculate and log perplexity scores.

        Args:
            func (Callable): Function to decorate

        Returns:
            Callable: Decorated function
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


class ResponseLength(Loggable):
    """Measures response length in tokens, words, and characters."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        text = result.choices[0].message.content

                        # Get completion tokens from usage if available
                        completion_tokens = getattr(
                            result.usage,
                            "completion_tokens",
                            len(text.split()),  # Fallback to word count
                        )

                        # Calculate various length metrics
                        metrics = {
                            "token_count": completion_tokens,
                            "word_count": len(text.split()),
                            "char_count": len(text),
                            "char_count_no_spaces": len(text.replace(" ", "")),
                        }

                        self._write_log(
                            "response_length",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                **metrics,
                                "success": True,
                            },
                            metric_category="generative",
                        )
                    except AttributeError as e:
                        self._write_log(
                            "response_length",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": f"Could not extract text: {str(e)}",
                            },
                            metric_category="generative",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "response_length",
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


class Repetition(Loggable):
    """Measures various repetition metrics in the text."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        ngram_ranges: list[int] = [1, 2, 3],  # n-gram sizes to check
        **kwargs,
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )
        self.ngram_ranges = ngram_ranges

    def _get_ngrams(self, text: str, n: int) -> list[str]:
        """Extract n-grams from text."""
        words = text.split()
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    def _calculate_repetition_metrics(self, text: str) -> dict:
        """Calculate various repetition metrics."""
        metrics = {}

        for n in self.ngram_ranges:
            ngrams = self._get_ngrams(text.lower(), n)
            if not ngrams:
                continue

            # Count occurrences
            ngram_counts = Counter(ngrams)

            # Calculate metrics
            unique_ratio = len(set(ngrams)) / len(ngrams)
            repetition_ratio = 1 - unique_ratio

            # Find most repeated n-gram and its count
            if ngram_counts:
                most_common = ngram_counts.most_common(1)[0]
                most_repeated = {
                    f"most_repeated_{n}gram": most_common[0],
                    f"most_repeated_{n}gram_count": most_common[1],
                }
            else:
                most_repeated = {
                    f"most_repeated_{n}gram": None,
                    f"most_repeated_{n}gram_count": 0,
                }

            metrics.update(
                {
                    f"{n}gram_unique_ratio": unique_ratio,
                    f"{n}gram_repetition_ratio": repetition_ratio,
                    **most_repeated,
                }
            )

        return metrics

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        text = result.choices[0].message.content
                        metrics = self._calculate_repetition_metrics(text)

                        self._write_log(
                            "repetition",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                **metrics,
                                "success": True,
                            },
                            metric_category="generative",
                        )
                    except AttributeError as e:
                        self._write_log(
                            "repetition",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": f"Could not extract text: {str(e)}",
                            },
                            metric_category="generative",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "repetition",
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


class VocabularyDiversity(Loggable):
    """Measures vocabulary diversity using multiple metrics."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def _calculate_diversity_metrics(self, text: str) -> dict:
        """Calculate various vocabulary diversity metrics."""
        # Tokenize
        words = text.lower().split()
        unique_words = set(words)

        if not words:
            return {
                "type_token_ratio": 0,
                "unique_word_count": 0,
                "total_word_count": 0,
                "hapax_legomena_ratio": 0,
            }

        # Calculate frequencies
        word_freq = Counter(words)
        hapax_legomena = sum(1 for word, freq in word_freq.items() if freq == 1)

        metrics = {
            "type_token_ratio": len(unique_words) / len(words),
            "unique_word_count": len(unique_words),
            "total_word_count": len(words),
            "hapax_legomena_ratio": hapax_legomena / len(words),
        }

        return metrics

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        text = result.choices[0].message.content
                        metrics = self._calculate_diversity_metrics(text)

                        self._write_log(
                            "vocabulary_diversity",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                **metrics,
                                "success": True,
                            },
                            metric_category="linguistic",
                        )
                    except AttributeError as e:
                        self._write_log(
                            "vocabulary_diversity",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": f"Could not extract text: {str(e)}",
                            },
                            metric_category="linguistic",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "vocabulary_diversity",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        metric_category="linguistic",
                    )
                raise

        return wrapper


class SentenceLength(Loggable):
    """Measures various sentence length metrics."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    def _calculate_sentence_metrics(self, text: str) -> dict:
        """Calculate various sentence length metrics."""
        # Use NLTK for sentence tokenization
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            return {
                "avg_sentence_length_words": 0,
                "avg_sentence_length_chars": 0,
                "max_sentence_length_words": 0,
                "min_sentence_length_words": 0,
                "sentence_length_std_words": 0,
                "sentence_count": 0,
            }

        # Calculate word counts per sentence
        word_counts = [len(sent.split()) for sent in sentences]
        char_counts = [len(sent) for sent in sentences]

        metrics = {
            "avg_sentence_length_words": np.mean(word_counts),
            "avg_sentence_length_chars": np.mean(char_counts),
            "max_sentence_length_words": max(word_counts),
            "min_sentence_length_words": min(word_counts),
            "sentence_length_std_words": np.std(word_counts),
            "sentence_count": len(sentences),
        }

        return metrics

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)

                if self.log:
                    try:
                        text = result.choices[0].message.content
                        metrics = self._calculate_sentence_metrics(text)

                        self._write_log(
                            "sentence_length",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                **metrics,
                                "success": True,
                            },
                            metric_category="linguistic",
                        )
                    except AttributeError as e:
                        self._write_log(
                            "sentence_length",
                            {
                                "timestamp": datetime.now().isoformat(),
                                "function": func.__name__,
                                "module": func.__module__,
                                "success": True,
                                "note": f"Could not extract text: {str(e)}",
                            },
                            metric_category="linguistic",
                        )

                return result

            except Exception as e:
                if self.log:
                    self._write_log(
                        "sentence_length",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "module": func.__module__,
                            "success": False,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        metric_category="linguistic",
                    )
                raise

        return wrapper


class GenerativeMetrics(Metrics):
    """Factory class for generative metrics with list-style decoration."""

    def __init__(
        self,
        log: bool = True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        **kwargs,
    ):
        """Initialize the GenerativeMetrics factory.

        Args:
            log (bool, optional): To log or not to log? Defaults to True.
            output_dir (str, optional): Where to log? Defaults to "./logs".
            experiment_name (str, optional): What to log? Defaults to "experiment".
        """
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

        # Initialize basic metrics
        self._perplexity = Perplexity(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )
        self._response_length = ResponseLength(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )
        self._repetition = Repetition(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )
        self._vocabulary_diversity = VocabularyDiversity(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )
        self._sentence_length = SentenceLength(
            log=log, output_dir=output_dir, experiment_name=experiment_name
        )

        # Mapping of metric names to their implementations
        self._metrics = {
            "perplexity": lambda: self._perplexity,
            "response_length": lambda: self._response_length,
            "repetition": lambda: self._repetition,
            "vocabulary_diversity": lambda: self._vocabulary_diversity,
            "sentence_length": lambda: self._sentence_length,
        }
