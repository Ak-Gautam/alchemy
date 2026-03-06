"""Dataset recipe interfaces and built-in recipes."""

from alchemy.recipes.base import BaseRecipe, LanguageConstraints
from alchemy.recipes.text_compression import TextCompressionPairsRecipe

_BUILTIN_RECIPES: dict[str, type[BaseRecipe]] = {
    "text_compression_pairs": TextCompressionPairsRecipe,
}


def list_recipe_names() -> list[str]:
    """Return sorted built-in recipe names."""
    return sorted(_BUILTIN_RECIPES)


def get_recipe(name: str) -> BaseRecipe:
    """Instantiate a built-in recipe by name."""
    cls = _BUILTIN_RECIPES.get(name)
    if cls is None:
        raise ValueError(f"Unknown recipe: {name!r}. Available: {sorted(_BUILTIN_RECIPES)}")
    return cls()


__all__ = [
    "BaseRecipe",
    "LanguageConstraints",
    "TextCompressionPairsRecipe",
    "get_recipe",
    "list_recipe_names",
]
