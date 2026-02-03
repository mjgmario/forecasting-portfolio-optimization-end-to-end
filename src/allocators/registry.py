"""
Allocator Registry

This module provides a centralized registry for all portfolio allocation methods using
the Registry pattern. Allocators can be registered automatically using decorators
or manually via the register() method.
"""

import logging
from typing import Any, Optional

from .base import BaseAllocator

logger = logging.getLogger(__name__)


class AllocatorRegistry:
    """
    Singleton registry for all allocators.

    Provides centralized access to all registered portfolio allocation methods.
    Uses the Registry pattern to enable plugin-style architecture.
    """

    _instance: Optional["AllocatorRegistry"] = None
    _allocators: dict[str, type[BaseAllocator]] = {}

    def __new__(cls) -> "AllocatorRegistry":
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls, name: str, allocator_class: type[BaseAllocator], override: bool = False
    ) -> None:
        """
        Register an allocator class.

        Args:
            name: Unique identifier for the allocator (e.g., "markowitz", "equal_weight")
            allocator_class: Class inheriting from BaseAllocator
            override: If True, allow overwriting existing registrations

        Raises:
            ValueError: If name already registered and override=False
            TypeError: If allocator_class is not a BaseAllocator subclass
        """
        # Validate allocator_class
        if not isinstance(allocator_class, type):
            raise TypeError(f"allocator_class must be a class, got {type(allocator_class)}")

        if not issubclass(allocator_class, BaseAllocator):
            raise TypeError(f"{allocator_class.__name__} must inherit from BaseAllocator")

        # Check for duplicate registration
        if name in cls._allocators and not override:
            raise ValueError(
                f"Allocator '{name}' is already registered. " f"Use override=True to replace."
            )

        cls._allocators[name] = allocator_class
        logger.debug(f"Registered allocator: {name} ({allocator_class.__name__})")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove an allocator from the registry.

        Args:
            name: Name of the allocator to remove

        Raises:
            KeyError: If allocator not found
        """
        if name not in cls._allocators:
            raise KeyError(f"Allocator '{name}' not found in registry")

        del cls._allocators[name]
        logger.debug(f"Unregistered allocator: {name}")

    @classmethod
    def get(cls, name: str) -> type[BaseAllocator]:
        """
        Get an allocator class by name.

        Args:
            name: Name of the allocator

        Returns:
            Allocator class (not instantiated)

        Raises:
            KeyError: If allocator not found

        Example:
            >>> allocator_class = AllocatorRegistry.get("markowitz")
            >>> allocator = allocator_class(config={"risk_aversion": 5})
            >>> weights = allocator.allocate(predicted_returns, historical_returns)
        """
        if name not in cls._allocators:
            available = ", ".join(cls.list_all())
            raise KeyError(
                f"Allocator '{name}' not found in registry. " f"Available allocators: {available}"
            )

        return cls._allocators[name]

    @classmethod
    def create(cls, name: str, config: dict | None = None) -> BaseAllocator:
        """
        Create and return an instance of an allocator.

        Args:
            name: Name of the allocator
            config: Optional configuration dictionary

        Returns:
            Instantiated allocator object

        Raises:
            KeyError: If allocator not found

        Example:
            >>> allocator = AllocatorRegistry.create("markowitz", {"risk_aversion": 5})
            >>> weights = allocator.allocate(predicted_returns, historical_returns)
        """
        allocator_class = cls.get(name)
        return allocator_class(config=config)

    @classmethod
    def list_all(cls) -> list[str]:
        """
        List all registered allocator names.

        Returns:
            List of allocator names sorted alphabetically

        Example:
            >>> AllocatorRegistry.list_all()
            ['equal_weight', 'markowitz', 'risk_parity']
        """
        return sorted(cls._allocators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an allocator is registered.

        Args:
            name: Allocator name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._allocators

    @classmethod
    def count(cls) -> int:
        """
        Get the total number of registered allocators.

        Returns:
            Count of registered allocators
        """
        return len(cls._allocators)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered allocators.

        Warning: This removes all allocators from the registry.
        Primarily useful for testing.
        """
        cls._allocators.clear()
        logger.debug("Cleared all allocators from registry")

    @classmethod
    def get_info(cls, name: str) -> dict[str, Any]:
        """
        Get information about a registered allocator.

        Args:
            name: Allocator name

        Returns:
            Dictionary with allocator metadata

        Raises:
            KeyError: If allocator not found
        """
        allocator_class = cls.get(name)
        return {
            "name": name,
            "class_name": allocator_class.__name__,
            "module": allocator_class.__module__,
            "docstring": allocator_class.__doc__,
        }

    @classmethod
    def list_all_info(cls) -> list[dict[str, Any]]:
        """
        Get detailed information about all registered allocators.

        Returns:
            List of dictionaries with allocator metadata
        """
        return [cls.get_info(name) for name in cls.list_all()]


def register_allocator(name: str, override: bool = False):
    """
    Decorator for automatic allocator registration.

    Args:
        name: Unique identifier for the allocator
        override: If True, allow overwriting existing registrations

    Returns:
        Decorator function

    Example:
        >>> @register_allocator("equal_weight")
        >>> class EqualWeightAllocator(BaseAllocator):
        >>>     # ... implementation
    """

    def decorator(allocator_class: type[BaseAllocator]) -> type[BaseAllocator]:
        AllocatorRegistry.register(name, allocator_class, override=override)
        return allocator_class

    return decorator


# Convenience function for easy importing
def get_allocator(name: str) -> type[BaseAllocator]:
    """
    Convenience function to get an allocator class.

    Args:
        name: Allocator name

    Returns:
        Allocator class

    Example:
        >>> from src.allocators import get_allocator
        >>> MarkowitzAllocator = get_allocator("markowitz")
    """
    return AllocatorRegistry.get(name)
