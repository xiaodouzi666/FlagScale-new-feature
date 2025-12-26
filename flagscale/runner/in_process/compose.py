"""Compose utility for chaining callable objects.

This module provides a Compose class for functional composition of
callable objects, inspired by NVIDIA's nvidia-resiliency-ext compose.
"""

import logging

from typing import Any, Callable, List, Tuple, Type

logger = logging.getLogger(__name__)


def find_common_ancestor(*instances) -> Type:
    """Find the lowest common ancestor class of multiple instances.

    Determines the most specific class that all instances inherit from
    by computing the intersection of their method resolution orders (MRO).

    Args:
        *instances: Variable number of class instances

    Returns:
        The lowest common ancestor class
    """
    if not instances:
        return object

    if len(instances) == 1:
        return type(instances[0])

    # Get MRO for each instance's class
    mros = [type(inst).__mro__ for inst in instances]

    # Find intersection of all MROs
    common = set(mros[0])
    for mro in mros[1:]:
        common &= set(mro)

    # Return the most specific common class (first in any MRO)
    for cls in mros[0]:
        if cls in common:
            return cls

    return object


class Compose:
    """Chains multiple callable objects together.

    The output of each callable is passed as input to the next callable.
    Execution order is from first to last in the provided list.

    Example:
        >>> def add_one(x): return x + 1
        >>> def double(x): return x * 2
        >>> composed = Compose([add_one, double])
        >>> composed(5)  # (5 + 1) * 2 = 12
        12
    """

    def __init__(
        self,
        callables: List[Callable],
        name: str = None,
    ):
        """Initialize the compose chain.

        Args:
            callables: List of callable objects to chain
            name: Optional name for this composition
        """
        self.callables = list(callables)
        self.name = name or "Compose"

        if not self.callables:
            logger.warning("Compose initialized with empty callable list")

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the composed callables.

        Args:
            *args: Arguments to pass to the first callable
            **kwargs: Keyword arguments to pass to the first callable

        Returns:
            Result of the final callable in the chain
        """
        if not self.callables:
            return args[0] if args else None

        # Execute first callable with original arguments
        result = self.callables[0](*args, **kwargs)

        # Chain through remaining callables
        for callable_obj in self.callables[1:]:
            try:
                # Handle tuple results
                if isinstance(result, tuple):
                    result = callable_obj(*result)
                else:
                    result = callable_obj(result)
            except Exception as e:
                logger.warning(
                    f"Error in compose chain at {callable_obj}: {e}"
                )
                raise

        return result

    def append(self, callable_obj: Callable) -> "Compose":
        """Append a callable to the chain.

        Args:
            callable_obj: Callable to append

        Returns:
            Self for chaining
        """
        self.callables.append(callable_obj)
        return self

    def prepend(self, callable_obj: Callable) -> "Compose":
        """Prepend a callable to the chain.

        Args:
            callable_obj: Callable to prepend

        Returns:
            Self for chaining
        """
        self.callables.insert(0, callable_obj)
        return self

    def __len__(self) -> int:
        """Return number of callables in chain."""
        return len(self.callables)

    def __repr__(self) -> str:
        """Return string representation."""
        callable_names = [
            getattr(c, "__name__", type(c).__name__)
            for c in self.callables
        ]
        return f"Compose({', '.join(callable_names)})"


class TypedCompose(Compose):
    """Compose with common ancestor type inheritance.

    Creates a dynamic class that inherits from both Compose and the
    common ancestor of all provided instances.
    """

    def __new__(cls, instances: List[Any], name: str = None):
        """Create a new TypedCompose with dynamic inheritance.

        Args:
            instances: List of instances to compose
            name: Optional name for this composition

        Returns:
            New TypedCompose instance
        """
        if not instances:
            return super().__new__(cls)

        # Find common ancestor
        ancestor = find_common_ancestor(*instances)

        # Create dynamic class inheriting from both
        class DynamicCompose(cls, ancestor):
            pass

        DynamicCompose.__name__ = name or f"TypedCompose[{ancestor.__name__}]"

        instance = object.__new__(DynamicCompose)
        return instance

    def __init__(self, instances: List[Any], name: str = None):
        """Initialize typed compose.

        Args:
            instances: List of instances to compose
            name: Optional name for this composition
        """
        # Extract callables from instances if they're callable
        callables = [inst for inst in instances if callable(inst)]
        super().__init__(callables, name)
        self.instances = instances


class Pipeline:
    """A more flexible pipeline for processing.

    Unlike Compose which passes results between callables, Pipeline
    allows for more control over how data flows through the chain.
    """

    def __init__(
        self,
        steps: List[Tuple[str, Callable]],
        error_handler: Callable[[str, Exception], Any] = None,
    ):
        """Initialize the pipeline.

        Args:
            steps: List of (name, callable) tuples
            error_handler: Optional handler for errors in steps
        """
        self.steps = steps
        self.error_handler = error_handler
        self._results = {}

    def run(self, initial_input: Any) -> Any:
        """Run the pipeline.

        Args:
            initial_input: Input to the first step

        Returns:
            Result of the final step
        """
        result = initial_input
        self._results = {}

        for name, step in self.steps:
            try:
                result = step(result)
                self._results[name] = result
                logger.debug(f"Pipeline step '{name}' completed")
            except Exception as e:
                logger.error(f"Pipeline step '{name}' failed: {e}")
                if self.error_handler:
                    result = self.error_handler(name, e)
                    self._results[name] = result
                else:
                    raise

        return result

    def get_result(self, step_name: str) -> Any:
        """Get the result of a specific step.

        Args:
            step_name: Name of the step

        Returns:
            Result of that step
        """
        return self._results.get(step_name)

    def get_all_results(self) -> dict:
        """Get results of all steps."""
        return self._results.copy()


def compose(*callables: Callable) -> Compose:
    """Convenience function for creating Compose objects.

    Args:
        *callables: Callables to compose

    Returns:
        Compose object
    """
    return Compose(list(callables))


def pipeline(*steps: Tuple[str, Callable]) -> Pipeline:
    """Convenience function for creating Pipeline objects.

    Args:
        *steps: (name, callable) tuples

    Returns:
        Pipeline object
    """
    return Pipeline(list(steps))
