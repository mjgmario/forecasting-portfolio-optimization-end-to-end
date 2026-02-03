"""
Forecaster Registry

This module provides a centralized registry for all forecasting models using
the Registry pattern. Forecasters can be registered automatically using decorators
or manually via the register() method.
"""

import logging
from typing import Any, Optional

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class ForecasterRegistry:
    """
    Singleton registry for all forecasters.

    Provides centralized access to all registered forecasting models.
    Uses the Registry pattern to enable plugin-style architecture.
    """

    _instance: Optional["ForecasterRegistry"] = None
    _forecasters: dict[str, type[BaseForecaster]] = {}

    def __new__(cls) -> "ForecasterRegistry":
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls, name: str, forecaster_class: type[BaseForecaster], override: bool = False
    ) -> None:
        """
        Register a forecaster class.

        Args:
            name: Unique identifier for the forecaster (e.g., "prophet", "arima")
            forecaster_class: Class inheriting from BaseForecaster
            override: If True, allow overwriting existing registrations

        Raises:
            ValueError: If name already registered and override=False
            TypeError: If forecaster_class is not a BaseForecaster subclass
        """
        # Validate forecaster_class
        if not isinstance(forecaster_class, type):
            raise TypeError(f"forecaster_class must be a class, got {type(forecaster_class)}")

        if not issubclass(forecaster_class, BaseForecaster):
            raise TypeError(f"{forecaster_class.__name__} must inherit from BaseForecaster")

        # Check for duplicate registration
        if name in cls._forecasters and not override:
            # If same class name, skip silently (handles re-imports during testing)
            existing_class = cls._forecasters[name]
            if existing_class.__name__ == forecaster_class.__name__:
                return
            raise ValueError(
                f"Forecaster '{name}' is already registered. " f"Use override=True to replace."
            )

        cls._forecasters[name] = forecaster_class
        logger.debug(f"Registered forecaster: {name} ({forecaster_class.__name__})")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a forecaster from the registry.

        Args:
            name: Name of the forecaster to remove

        Raises:
            KeyError: If forecaster not found
        """
        if name not in cls._forecasters:
            raise KeyError(f"Forecaster '{name}' not found in registry")

        del cls._forecasters[name]
        logger.debug(f"Unregistered forecaster: {name}")

    @classmethod
    def get(cls, name: str) -> type[BaseForecaster]:
        """
        Get a forecaster class by name.

        Args:
            name: Name of the forecaster

        Returns:
            Forecaster class (not instantiated)

        Raises:
            KeyError: If forecaster not found

        Example:
            >>> forecaster_class = ForecasterRegistry.get("prophet")
            >>> forecaster = forecaster_class(config={"yearly_seasonality": True})
            >>> forecaster.fit(train_data)
        """
        if name not in cls._forecasters:
            available = ", ".join(cls.list_all())
            raise KeyError(
                f"Forecaster '{name}' not found in registry. " f"Available forecasters: {available}"
            )

        return cls._forecasters[name]

    @classmethod
    def create(cls, name: str, config: dict | None = None) -> BaseForecaster:
        """
        Create and return an instance of a forecaster.

        Args:
            name: Name of the forecaster
            config: Optional configuration dictionary

        Returns:
            Instantiated forecaster object

        Raises:
            KeyError: If forecaster not found

        Example:
            >>> forecaster = ForecasterRegistry.create("prophet", {"yearly_seasonality": True})
            >>> forecaster.fit(train_data)
        """
        forecaster_class = cls.get(name)
        return forecaster_class(config=config)

    @classmethod
    def list_all(cls) -> list[str]:
        """
        List all registered forecaster names.

        Returns:
            List of forecaster names sorted alphabetically

        Example:
            >>> ForecasterRegistry.list_all()
            ['arima', 'naive', 'prophet', 'xgboost']
        """
        return sorted(cls._forecasters.keys())

    @classmethod
    def list_by_family(cls, family: str) -> list[str]:
        """
        List forecasters belonging to a specific family.

        Args:
            family: Family name (e.g., "baseline", "statistical", "ml", "dl")

        Returns:
            List of forecaster names in the specified family

        Example:
            >>> ForecasterRegistry.list_by_family("baseline")
            ['naive', 'seasonal_naive', 'drift']
        """
        result = []
        for name, forecaster_class in cls._forecasters.items():
            # Check if class has family attribute
            if hasattr(forecaster_class, "family"):
                if forecaster_class.family == family:
                    result.append(name)
        return sorted(result)

    @classmethod
    def get_families(cls) -> list[str]:
        """
        Get list of all unique forecaster families.

        Returns:
            List of unique family names

        Example:
            >>> ForecasterRegistry.get_families()
            ['baseline', 'dl', 'ml', 'statistical']
        """
        families = set()
        for forecaster_class in cls._forecasters.values():
            if hasattr(forecaster_class, "family"):
                families.add(forecaster_class.family)
        return sorted(families)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a forecaster is registered.

        Args:
            name: Forecaster name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._forecasters

    @classmethod
    def count(cls) -> int:
        """
        Get the total number of registered forecasters.

        Returns:
            Count of registered forecasters
        """
        return len(cls._forecasters)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered forecasters.

        Warning: This removes all forecasters from the registry.
        Primarily useful for testing.
        """
        cls._forecasters.clear()
        logger.debug("Cleared all forecasters from registry")

    @classmethod
    def get_info(cls, name: str) -> dict[str, Any]:
        """
        Get information about a registered forecaster.

        Args:
            name: Forecaster name

        Returns:
            Dictionary with forecaster metadata

        Raises:
            KeyError: If forecaster not found
        """
        forecaster_class = cls.get(name)
        return {
            "name": name,
            "class_name": forecaster_class.__name__,
            "family": getattr(forecaster_class, "family", "unknown"),
            "module": forecaster_class.__module__,
            "docstring": forecaster_class.__doc__,
        }

    @classmethod
    def list_all_info(cls) -> list[dict[str, Any]]:
        """
        Get detailed information about all registered forecasters.

        Returns:
            List of dictionaries with forecaster metadata
        """
        return [cls.get_info(name) for name in cls.list_all()]


def register_forecaster(name: str, override: bool = False):
    """
    Decorator for automatic forecaster registration.

    Args:
        name: Unique identifier for the forecaster
        override: If True, allow overwriting existing registrations

    Returns:
        Decorator function

    Example:
        >>> @register_forecaster("naive")
        >>> class NaiveForecaster(BaseForecaster):
        >>>     family = "baseline"
        >>>     # ... implementation
    """

    def decorator(forecaster_class: type[BaseForecaster]) -> type[BaseForecaster]:
        ForecasterRegistry.register(name, forecaster_class, override=override)
        return forecaster_class

    return decorator


# Convenience function for easy importing
def get_forecaster(name: str) -> type[BaseForecaster]:
    """
    Convenience function to get a forecaster class.

    Args:
        name: Forecaster name

    Returns:
        Forecaster class

    Example:
        >>> from src.forecasters import get_forecaster
        >>> ProphetForecaster = get_forecaster("prophet")
    """
    return ForecasterRegistry.get(name)
