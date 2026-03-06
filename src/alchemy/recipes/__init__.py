"""Dataset recipe interfaces and built-in recipes."""

from alchemy.recipes.base import BaseRecipe, LanguageConstraints
from alchemy.recipes.text_compression import TextCompressionPairsRecipe

_BUILTIN_RECIPES: dict[str, type[BaseRecipe]] = {
    "text_compression_pairs": TextCompressionPairsRecipe,
}


def get_recipe(name: str) -> BaseRecipe:
    """Instantiate a built-in recipe by name."""
    cls = _BUILTIN_RECIPES.get(name)
    if cls is None:
        raise ValueError(f"Unknown recipe: {name!r}. Available: {sorted(_BUILTIN_RECIPES)}")
    return cls()


__all__ = ["BaseRecipe", "LanguageConstraints", "TextCompressionPairsRecipe", "get_recipe"]
