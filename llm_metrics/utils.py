from typing import Callable


def combine_metrics(metric_list: list[Callable]) -> Callable:
    """Helper function to combine multiple metric decorators."""

    def decorator(func: Callable) -> Callable:
        for metric in reversed(metric_list):
            func = metric(func)
        return func

    return decorator
