"""Tests for text compression recipe."""

from __future__ import annotations

import pytest

from alchemy.recipes import TextCompressionPairsRecipe, get_recipe


def test_text_compression_recipe_builds_plan():
    recipe = TextCompressionPairsRecipe()
    plan = recipe.build_plan(num_rows=200, batch_size=20)
    assert plan.dataset_name == "text_compression_pairs"
    assert plan.defaults.num_rows == 200
    assert plan.defaults.batch_size == 20
    assert "source_text" in plan.row_schema["properties"]
    assert len(plan.variation_spec.axes) >= 2


def test_get_recipe_returns_built_in():
    recipe = get_recipe("text_compression_pairs")
    assert isinstance(recipe, TextCompressionPairsRecipe)


def test_get_recipe_unknown_raises():
    with pytest.raises(ValueError, match="Unknown recipe"):
        get_recipe("nonexistent_recipe")
