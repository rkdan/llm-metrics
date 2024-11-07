from typing import Callable


def combine_metrics(metric_list: list[Callable]) -> Callable:
    """Combine multiple metrics into a single decorator.

    Args:
        metric_list (list[Callable]): List of metric decorators.

    Returns:
        Callable: Combined decorator function.
    """

    def decorator(func: Callable) -> Callable:
        for metric in reversed(metric_list):
            func = metric(func)
        return func

    return decorator
