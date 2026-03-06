"""Shared test fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from alchemy.models.base import GenerationConfig, GenerationResult, Message, ModelProvider
from alchemy.pipeline.plan import FieldSchema, GenerationPlan
from alchemy.spec.plan import GenerationPlan as SchemaGenerationPlan


class MockProvider(ModelProvider):
    """A mock model provider that returns pre-configured responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_index = 0
        self.call_history: list[list[Message]] = []
        self.config_history: list[GenerationConfig | None] = []

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        self.call_history.append(messages)
        self.config_history.append(config)
        if self._call_index < len(self._responses):
            text = self._responses[self._call_index]
            self._call_index += 1
        else:
            text = "{}"
        return GenerationResult(text=text, model="mock-model")

    def model_name(self) -> str:
        return "mock-model"


@pytest.fixture
def mock_provider():
    """Create a MockProvider factory."""
    def _factory(responses: list[str] | None = None) -> MockProvider:
        return MockProvider(responses)
    return _factory


@pytest.fixture
def sample_plan() -> GenerationPlan:
    """A sample generation plan for testing."""
    return GenerationPlan(
        dataset_name="test_qa",
        description="Test question-answer pairs",
        fields=[
            FieldSchema(
                name="question",
                field_type="string",
                description="A test question",
                constraints={"min_length": 10, "max_length": 500},
            ),
            FieldSchema(
                name="answer",
                field_type="string",
                description="The answer",
                constraints={"min_length": 5},
            ),
            FieldSchema(
                name="difficulty",
                field_type="string",
                description="Difficulty level",
            ),
        ],
        generation_strategy="Vary topic and difficulty",
        num_samples=20,
        batch_size=5,
        example_samples=[
            {
                "question": "What is the capital of France?",
                "answer": "Paris is the capital of France.",
                "difficulty": "easy",
            }
        ],
        diversity_dimensions=["topic", "difficulty"],
        quality_criteria=["factual accuracy", "clarity"],
    )


@pytest.fixture
def sample_schema_plan() -> SchemaGenerationPlan:
    """A sample schema-first generation plan for testing."""
    return SchemaGenerationPlan(
        dataset_name="instruction_pairs",
        description="Instruction to completion pairs",
        row_schema={
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "minLength": 5},
                "completion": {"type": "string", "minLength": 5},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            },
            "required": ["instruction", "completion"],
            "additionalProperties": False,
        },
        defaults={"num_rows": 100, "batch_size": 10},
        variation_spec={
            "axes": [
                {
                    "name": "difficulty",
                    "values": ["easy", "medium", "hard"],
                    "distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                }
            ]
        },
        quality_rubric=["factuality", "clarity", "diversity"],
        example_rows=[
            {
                "instruction": "Explain recursion with a simple example.",
                "completion": "Recursion is a function calling itself...",
                "difficulty": "easy",
            }
        ],
        safety={"allow_pii": False, "disallowed_categories": ["self-harm"]},
    )
