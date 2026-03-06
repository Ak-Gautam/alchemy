"""Tests for planner agent."""

from __future__ import annotations

from alchemy.agents.planner import PlannerAgent


def test_planner_parses_valid_json(mock_provider, sample_plan):
    plan_json = sample_plan.model_dump_json()
    provider = mock_provider([plan_json])
    agent = PlannerAgent(provider)
    result = agent.plan("Generate test data")
    assert result.dataset_name == "test_qa"
    assert len(result.fields) == 3


def test_planner_handles_markdown_fences(mock_provider, sample_plan):
    plan_json = f"```json\n{sample_plan.model_dump_json()}\n```"
    provider = mock_provider([plan_json])
    agent = PlannerAgent(provider)
    result = agent.plan("Generate test data")
    assert result.dataset_name == "test_qa"


def test_planner_retries_on_bad_json(mock_provider, sample_plan):
    plan_json = sample_plan.model_dump_json()
    # First response is bad, second is good
    provider = mock_provider(["not json at all", plan_json])
    agent = PlannerAgent(provider, max_retries=3)
    result = agent.plan("Generate test data")
    assert result.dataset_name == "test_qa"
    # Should have called the provider twice (first fail + retry)
    assert len(provider.call_history) == 2
