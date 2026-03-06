"""Tests for schema planner agent."""

from __future__ import annotations

from alchemy.agents.schema_planner import SchemaPlannerAgent


def test_schema_planner_parses_valid_json(mock_provider, sample_schema_plan):
    plan_json = sample_schema_plan.model_dump_json()
    provider = mock_provider([plan_json])
    agent = SchemaPlannerAgent(provider)
    result = agent.plan("Generate instruction tuning rows")
    assert result.dataset_name == "instruction_pairs"
    assert result.row_schema["type"] == "object"
    assert result.defaults.num_rows == 100


def test_schema_planner_handles_markdown_fences(mock_provider, sample_schema_plan):
    plan_json = f"```json\n{sample_schema_plan.model_dump_json()}\n```"
    provider = mock_provider([plan_json])
    agent = SchemaPlannerAgent(provider)
    result = agent.plan("Generate instruction tuning rows")
    assert result.dataset_name == "instruction_pairs"


def test_schema_planner_retries_on_bad_json(mock_provider, sample_schema_plan):
    plan_json = sample_schema_plan.model_dump_json()
    provider = mock_provider(["not json", plan_json])
    agent = SchemaPlannerAgent(provider, max_retries=3)
    result = agent.plan("Generate instruction tuning rows")
    assert result.dataset_name == "instruction_pairs"
    assert len(provider.call_history) == 2
