"""Initialization and retry control for restart loop.

This module provides initialization handlers and retry controllers
that determine whether restart attempts should continue.

Inspired by NVIDIA's nvidia-resiliency-ext initialize module.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .exception import RestartAbort
from .state import FrozenRankState, RankState

logger = logging.getLogger(__name__)


class Initialize(ABC):
    """Abstract base class for initialization handlers.

    Initialization handlers are called at the start of each restart
    iteration. They can:
    - Perform setup operations
    - Raise RestartAbort to terminate the restart loop
    - Raise Exception to trigger another restart attempt
    """

    @abstractmethod
    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Execute the initialization handler.

        Args:
            state: Current frozen rank state

        Returns:
            The state (possibly modified)

        Raises:
            RestartAbort: To terminate the restart loop
            Exception: To trigger another restart attempt
        """
        pass


class RetryController(Initialize):
    """Controls whether restart attempts should continue.

    This handler checks various conditions to determine if the
    restart loop should continue or abort.
    """

    def __init__(
        self,
        max_restarts: int = 3,
        min_world_size: int = 1,
        min_active_world_size: int = 1,
        restart_window: float = None,
        max_restarts_in_window: int = None,
    ):
        """Initialize the retry controller.

        Args:
            max_restarts: Maximum number of restart attempts (0 = unlimited)
            min_world_size: Minimum world size to continue
            min_active_world_size: Minimum active world size to continue
            restart_window: Time window in seconds for rate limiting
            max_restarts_in_window: Maximum restarts allowed in the window
        """
        self.max_restarts = max_restarts
        self.min_world_size = min_world_size
        self.min_active_world_size = min_active_world_size
        self.restart_window = restart_window
        self.max_restarts_in_window = max_restarts_in_window

        # Track restart timestamps for rate limiting
        self._restart_timestamps: List[float] = []

    def should_continue(self, state: FrozenRankState) -> bool:
        """Check if restart should continue.

        Args:
            state: Current frozen rank state

        Returns:
            True if restart should continue, False otherwise
        """
        # Check max restarts
        if self.max_restarts > 0 and state.restart_attempt >= self.max_restarts:
            logger.warning(
                f"Max restarts exceeded: {state.restart_attempt} >= {self.max_restarts}"
            )
            return False

        # Check world size
        if state.world_size < self.min_world_size:
            logger.warning(
                f"World size too small: {state.world_size} < {self.min_world_size}"
            )
            return False

        # Check active world size
        if state.active_world_size < self.min_active_world_size:
            logger.warning(
                f"Active world size too small: {state.active_world_size} < {self.min_active_world_size}"
            )
            return False

        # Check rate limiting
        if self.restart_window and self.max_restarts_in_window:
            now = time.time()
            # Remove old timestamps
            self._restart_timestamps = [
                ts for ts in self._restart_timestamps
                if now - ts < self.restart_window
            ]
            if len(self._restart_timestamps) >= self.max_restarts_in_window:
                logger.warning(
                    f"Restart rate limit exceeded: {len(self._restart_timestamps)} "
                    f"restarts in {self.restart_window}s window"
                )
                return False

        return True

    def record_restart(self) -> None:
        """Record a restart timestamp for rate limiting."""
        self._restart_timestamps.append(time.time())

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Check if restart should continue.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged

        Raises:
            RestartAbort: If restart should be terminated
        """
        if not self.should_continue(state):
            raise RestartAbort(
                reason=self._get_abort_reason(state),
                restart_count=state.restart_attempt,
            )

        # Record this restart attempt
        self.record_restart()

        logger.info(
            f"Restart attempt {state.restart_attempt + 1}"
            + (f"/{self.max_restarts}" if self.max_restarts > 0 else "")
            + f" for rank {state.rank}"
        )

        return state

    def _get_abort_reason(self, state: FrozenRankState) -> str:
        """Get the reason for aborting restart.

        Args:
            state: Current frozen rank state

        Returns:
            Description of why restart is being aborted
        """
        reasons = []

        if self.max_restarts > 0 and state.restart_attempt >= self.max_restarts:
            reasons.append(f"max restarts ({self.max_restarts}) exceeded")

        if state.world_size < self.min_world_size:
            reasons.append(
                f"world size ({state.world_size}) below minimum ({self.min_world_size})"
            )

        if state.active_world_size < self.min_active_world_size:
            reasons.append(
                f"active world size ({state.active_world_size}) below minimum ({self.min_active_world_size})"
            )

        if self.restart_window and self.max_restarts_in_window:
            recent = len(self._restart_timestamps)
            if recent >= self.max_restarts_in_window:
                reasons.append(
                    f"rate limit ({recent} restarts in {self.restart_window}s)"
                )

        return "; ".join(reasons) if reasons else "unknown"


class InitializeDistributed(Initialize):
    """Initialize PyTorch distributed after restart.

    This handler re-initializes torch.distributed with the
    current world configuration.
    """

    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = None,
        timeout_seconds: float = 300.0,
    ):
        """Initialize the handler.

        Args:
            backend: Distributed backend (nccl, gloo, etc.)
            init_method: Initialization method (env://, tcp://, etc.)
            timeout_seconds: Timeout for initialization
        """
        self.backend = backend
        self.init_method = init_method
        self.timeout_seconds = timeout_seconds

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Re-initialize PyTorch distributed.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged
        """
        try:
            import torch.distributed as dist
            from datetime import timedelta

            # Only initialize if not already initialized
            if not dist.is_initialized():
                logger.info(
                    f"Rank {state.rank}: Initializing distributed "
                    f"(backend={self.backend}, world_size={state.world_size})"
                )

                init_kwargs = {
                    "backend": self.backend,
                    "world_size": state.world_size,
                    "rank": state.rank,
                    "timeout": timedelta(seconds=self.timeout_seconds),
                }

                if self.init_method:
                    init_kwargs["init_method"] = self.init_method

                dist.init_process_group(**init_kwargs)

                logger.info(f"Rank {state.rank}: Distributed initialized successfully")

        except ImportError:
            logger.debug("torch.distributed not available")
        except Exception as e:
            logger.error(f"Rank {state.rank}: Failed to initialize distributed: {e}")
            raise

        return state


class ComposedInitialize(Initialize):
    """Compose multiple initialization handlers.

    Executes initialization handlers in sequence.
    """

    def __init__(self, handlers: List[Initialize]):
        """Initialize with a list of handlers.

        Args:
            handlers: List of initialization handlers to execute in order
        """
        self.handlers = handlers

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Execute all initialization handlers in sequence.

        Args:
            state: Current frozen rank state

        Returns:
            The state after all handlers have executed
        """
        for handler in self.handlers:
            state = handler(state)
        return state


@dataclass
class RestartConfig:
    """Configuration for restart behavior.

    Attributes:
        max_restarts: Maximum number of restart attempts (0 = unlimited)
        min_world_size: Minimum world size to continue
        restart_on_health_check_fail: Restart when health check fails
        restart_on_heartbeat_timeout: Restart when heartbeat times out
        restart_on_hang: Restart when hang is detected
        restart_delay: Delay in seconds between restart attempts
        exponential_backoff: Use exponential backoff for restart delays
        max_restart_delay: Maximum delay between restarts
    """

    max_restarts: int = 3
    min_world_size: int = 1
    restart_on_health_check_fail: bool = True
    restart_on_heartbeat_timeout: bool = True
    restart_on_hang: bool = True
    restart_delay: float = 1.0
    exponential_backoff: bool = True
    max_restart_delay: float = 60.0

    def get_delay(self, restart_attempt: int) -> float:
        """Get the delay for the given restart attempt.

        Args:
            restart_attempt: Current restart attempt

        Returns:
            Delay in seconds
        """
        if not self.exponential_backoff:
            return self.restart_delay

        delay = self.restart_delay * (2 ** restart_attempt)
        return min(delay, self.max_restart_delay)


def create_default_retry_controller(
    max_restarts: int = 3,
    min_world_size: int = 1,
) -> RetryController:
    """Create a default retry controller.

    Args:
        max_restarts: Maximum number of restart attempts
        min_world_size: Minimum world size to continue

    Returns:
        A configured RetryController
    """
    return RetryController(
        max_restarts=max_restarts,
        min_world_size=min_world_size,
        min_active_world_size=1,
    )
