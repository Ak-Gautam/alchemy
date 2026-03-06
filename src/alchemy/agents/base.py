"""Base agent with retry and self-correction logic."""

from __future__ import annotations

import abc
import logging
from typing import Any

from alchemy.models.base import GenerationConfig, GenerationResult, Message, ModelProvider

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Base class for all agent roles.

    Wraps a ModelProvider with role-specific system prompts,
    output parsing, and a retry loop that feeds parse errors
    back to the model for self-correction.
    """

    def __init__(
        self,
        provider: ModelProvider,
        generation_config: GenerationConfig | None = None,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.generation_config = generation_config or GenerationConfig()
        self.max_retries = max_retries

    @abc.abstractmethod
    def system_prompt(self, **kwargs: Any) -> str:
        ...

    @abc.abstractmethod
    def parse_response(self, result: GenerationResult) -> Any:
        ...

    def invoke(self, user_message: str, **prompt_kwargs: Any) -> Any:
        """Send a message and parse the response, retrying on parse failures."""
        messages = [
            Message(role="system", content=self.system_prompt(**prompt_kwargs)),
            Message(role="user", content=user_message),
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            result = self.provider.generate(messages, self.generation_config)
            try:
                return self.parse_response(result)
            except Exception as e:
                logger.warning(
                    "Agent parse failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                last_error = e
                messages.append(Message(role="assistant", content=result.text))
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Your previous response could not be parsed: {e}. "
                            "Please try again with valid JSON output only."
                        ),
                    )
                )

        raise last_error  # type: ignore[misc]
