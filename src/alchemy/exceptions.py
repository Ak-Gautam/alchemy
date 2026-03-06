"""Custom exceptions for the alchemy package."""

from __future__ import annotations


class AlchemyError(Exception):
    """Base exception for all alchemy errors."""

    def __init__(self, message: str, details: str = ""):
        self.details = details
        super().__init__(message)


class ModelError(AlchemyError):
    """Raised when a model provider fails."""


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded."""


class PlanningError(AlchemyError):
    """Raised when the planner agent fails to produce a valid plan."""


class GenerationError(AlchemyError):
    """Raised when the generator agent fails to produce valid samples."""


class ValidationError(AlchemyError):
    """Raised when the validator agent encounters an unrecoverable error."""


class ConfigError(AlchemyError):
    """Raised when configuration is invalid."""


class OutputError(AlchemyError):
    """Raised when writing output fails."""
