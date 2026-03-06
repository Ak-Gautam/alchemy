"""Recipe for text compression training pairs."""

from __future__ import annotations

from alchemy.spec.plan import VariationAxis, VariationSpec

from .base import BaseRecipe


class TextCompressionPairsRecipe(BaseRecipe):
    """Builds plans for source/compressed text pair datasets."""

    @property
    def name(self) -> str:
        return "text_compression_pairs"

    @property
    def description(self) -> str:
        return (
            "Dataset of source texts and compact rewrites preserving key information. "
            "Useful for compression-aware summarization and instruction tuning."
        )

    @property
    def row_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_text": {"type": "string", "minLength": 30},
                "compressed_text": {"type": "string", "minLength": 10},
                "compression_style": {
                    "type": "string",
                    "enum": ["extractive", "abstractive", "bullet", "telegraphic"],
                },
                "target_ratio": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                "domain": {
                    "type": "string",
                    "enum": ["science", "news", "legal", "technical", "general"],
                },
            },
            "required": [
                "source_text",
                "compressed_text",
                "compression_style",
                "target_ratio",
                "domain",
            ],
            "additionalProperties": False,
        }

    @property
    def quality_rubric(self) -> list[str]:
        return [
            "Preserves key facts from source_text",
            "compressed_text is significantly shorter and coherent",
            "compression_style matches requested style",
            "No fabricated claims or contradictions",
        ]

    @property
    def variation_spec(self) -> VariationSpec:
        return VariationSpec(
            axes=[
                VariationAxis(
                    name="compression_style",
                    values=["extractive", "abstractive", "bullet", "telegraphic"],
                    distribution={
                        "extractive": 0.25,
                        "abstractive": 0.35,
                        "bullet": 0.2,
                        "telegraphic": 0.2,
                    },
                ),
                VariationAxis(
                    name="domain",
                    values=["science", "news", "legal", "technical", "general"],
                ),
            ],
            constraints={"max_duplicate_source_per_100": 2},
        )
