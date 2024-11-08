import time
import uuid
from datetime import datetime
from typing import Any, Optional

import dotenv
import yaml
from openai import OpenAI

from .base import Loggable

dotenv.load_dotenv()


class OpenAIWrapper(Loggable):
    """A simple wrapper for the OpenAI API with usage tracking."""

    def __init__(
        self,
        log=True,
        output_dir: str = "./logs",
        experiment_name: str = "experiment",
        costs_dir: Optional[str] = None,
        **kwargs
    ):
        """Wrapper for the OpenAI API with usage tracking.

        Args:
            log (bool, optional): _description_. Defaults to True.
            output_dir (str, optional): _description_. Defaults to "./logs".
            experiment_name (str, optional): _description_. Defaults to "experiment".
            api_key (Optional[str], optional): _description_. Defaults to None.
            costs_dir (Optional[str], optional): _description_. Defaults to None.
        """
        super().__init__(
            log=log,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

        self.client = OpenAI()

        self.log = log

        if self.log:
            self.conversation_id = str(uuid.uuid4())

            if costs_dir:
                with open(costs_dir, "r") as f:
                    self.costs = yaml.safe_load(f)
            else:
                self.costs = {}

    def _log_metadata(self):
        """_summary_"""
        metadata = {
            "conversation_id": self.conversation_id,
            "session_start": self.session_start.isoformat(),
            "experiment_name": self.experiment_name,
        }
        self._write_log("metadata", metadata)

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> dict[str, float]:
        """_summary_

        Args:
            model (str): _description_
            prompt_tokens (int): _description_
            completion_tokens (int): _description_

        Returns:
            dict[str, float]: _description_
        """
        model_cost = self.costs.get(model, {"input": 0, "output": 0})
        input_cost = model_cost["input"] * prompt_tokens / 1000.0
        output_cost = model_cost["output"] * completion_tokens / 1000.0
        total_cost = input_cost + output_cost
        return {"input": input_cost, "output": output_cost, "total": total_cost}

    def completion(
        self, messages: list[dict[str, str]], model: str = "gpt-4o-mini", **kwargs
    ) -> Any:
        """Call the OpenAI API for completion and log the usage.
        All valid OpenAI parameters can be passed as keyword arguments.

        Args:
            messages (list[dict[str, str]]): List of messages to send to the model
            model (str, optional): _description_. Defaults to "gpt-4o-mini".

        Returns:
            Any: _description_
        """
        start_time = time.time()

        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

            if self.log:
                # Log request
                self._write_log(
                    "interactions",
                    {
                        "request_id": response.id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "parameters": kwargs,
                    },
                )

                costs = self._calculate_cost(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

                # Log usage
                self._write_log(
                    "token_usage",
                    {
                        "request_id": response.id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "duration_seconds": time.time() - start_time,
                        "costs": costs,
                    },
                )

                self._log_metadata()

            return response

        except Exception as e:
            # Log error and re-raise
            if self.log:
                self._write_log(
                    "errors",
                    {
                        "request_id": self.conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_seconds": time.time() - start_time,
                    },
                )
            raise
