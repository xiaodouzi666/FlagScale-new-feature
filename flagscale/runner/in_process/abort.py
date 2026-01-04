"""Abort handlers for cleaning up resources during restart.

This module provides abort handlers that clean up distributed training
resources when a fault is detected and restart is triggered.

Inspired by NVIDIA's nvidia-resiliency-ext abort module.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from .state import FrozenRankState

logger = logging.getLogger(__name__)


class Abort(ABC):
    """Abstract base class for abort handlers.

    Abort handlers are called when a fault is detected to clean up
    resources before restart. They receive the current state and
    return it (possibly modified).
    """

    @abstractmethod
    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Execute the abort handler.

        Args:
            state: Current frozen rank state

        Returns:
            The state (possibly modified)
        """
        pass


class AbortTorchDistributed(Abort):
    """Abort handler that destroys PyTorch distributed process groups.

    This handler cleans up torch.distributed resources to allow
    re-initialization during restart.
    """

    def __init__(self, timeout: float = 10.0):
        """Initialize the abort handler.

        Args:
            timeout: Timeout for cleanup operations in seconds
        """
        self.timeout = timeout

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Destroy PyTorch distributed process groups.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged
        """
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                logger.info(f"Rank {state.rank}: Destroying process groups...")

                # Try to destroy in a separate thread with timeout
                def destroy_pg():
                    try:
                        dist.destroy_process_group()
                    except Exception as e:
                        logger.warning(f"Error destroying process group: {e}")

                thread = threading.Thread(target=destroy_pg)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.timeout)

                if thread.is_alive():
                    logger.warning(
                        f"Rank {state.rank}: Process group destruction timed out"
                    )
                else:
                    logger.info(
                        f"Rank {state.rank}: Process groups destroyed successfully"
                    )

        except ImportError:
            logger.debug("torch.distributed not available")
        except Exception as e:
            logger.warning(f"Rank {state.rank}: Error during abort: {e}")

        return state


class AbortNCCL(Abort):
    """Abort handler for NCCL-specific cleanup.

    This handler attempts to abort NCCL operations which may be
    blocking due to a failed rank.
    """

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Abort NCCL operations.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged
        """
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                # Get the default process group
                pg = dist.distributed_c10d._get_default_group()
                if pg is not None:
                    # Try to get NCCL backend and abort
                    try:
                        backend = pg._get_backend(dist.get_backend())
                        if hasattr(backend, "abort"):
                            logger.info(f"Rank {state.rank}: Aborting NCCL backend...")
                            backend.abort()
                    except Exception as e:
                        logger.debug(f"Could not abort NCCL backend: {e}")

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Rank {state.rank}: Error aborting NCCL: {e}")

        return state


class AbortCUDA(Abort):
    """Abort handler for CUDA cleanup.

    This handler synchronizes and optionally resets CUDA state.
    """

    def __init__(self, reset_device: bool = False):
        """Initialize the abort handler.

        Args:
            reset_device: Whether to reset the CUDA device
        """
        self.reset_device = reset_device

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Clean up CUDA state.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged
        """
        try:
            import torch.cuda

            if torch.cuda.is_available():
                # Synchronize to ensure all operations complete
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.debug(f"CUDA synchronize failed: {e}")

                # Clear cache
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"CUDA empty_cache failed: {e}")

                # Optionally reset the device
                if self.reset_device:
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as e:
                        logger.debug(f"CUDA reset failed: {e}")

                logger.info(f"Rank {state.rank}: CUDA cleanup completed")

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Rank {state.rank}: Error during CUDA cleanup: {e}")

        return state


class AbortMegatron(Abort):
    """Abort handler for Megatron-LM state cleanup.

    This handler resets Megatron timers and other state that needs to be
    cleaned up for restart, while preserving args and other global state
    that the train function depends on.
    """

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Clean up Megatron state for restart.

        Args:
            state: Current frozen rank state

        Returns:
            The state unchanged
        """
        try:
            # Reset timers - this is critical to avoid "timer already started" errors
            try:
                from megatron.training.global_vars import get_timers
                timers = get_timers()
                if timers is not None:
                    # Reset all timer states
                    if hasattr(timers, '_timers'):
                        for name, timer in timers._timers.items():
                            # Reset the timer's internal state
                            if hasattr(timer, '_started'):
                                timer._started = False
                            if hasattr(timer, '_start_time'):
                                timer._start_time = None
                            if hasattr(timer, '_elapsed'):
                                timer._elapsed = 0.0
                            if hasattr(timer, '_elapsed_for_this_iteration'):
                                timer._elapsed_for_this_iteration = 0.0
                    logger.debug(f"Rank {state.rank}: Megatron timers reset")
            except Exception as e:
                logger.debug(f"Failed to reset timers: {e}")

            # Note: We intentionally do NOT call destroy_global_vars() because
            # it would destroy args which the train function needs

            logger.info(f"Rank {state.rank}: Megatron state cleanup completed")

        except ImportError:
            logger.debug("Megatron not available for cleanup")
        except Exception as e:
            logger.warning(f"Rank {state.rank}: Error during Megatron cleanup: {e}")

        return state


class ComposedAbort(Abort):
    """Compose multiple abort handlers.

    Executes abort handlers in sequence.
    """

    def __init__(self, handlers: List[Abort]):
        """Initialize with a list of handlers.

        Args:
            handlers: List of abort handlers to execute in order
        """
        self.handlers = handlers

    def __call__(self, state: FrozenRankState) -> FrozenRankState:
        """Execute all abort handlers in sequence.

        Args:
            state: Current frozen rank state

        Returns:
            The state after all handlers have executed
        """
        for handler in self.handlers:
            try:
                state = handler(state)
            except Exception as e:
                logger.warning(f"Abort handler {handler.__class__.__name__} failed: {e}")
        return state


def create_default_abort_handler() -> Abort:
    """Create a default abort handler with common cleanup operations.

    For in-process restart where we only wrap the train() function (not the
    full pretrain()), we should NOT destroy process groups or Megatron state
    because train() depends on them. We only reset timers to avoid
    "timer already started" errors.

    Returns:
        A composed abort handler with Megatron timer reset and CUDA cleanup
    """
    return ComposedAbort([
        AbortMegatron(),  # Reset timers only, preserve other state
        AbortCUDA(reset_device=False),
        # Note: We intentionally skip AbortNCCL() and AbortTorchDistributed()
        # because train() needs the process groups to be initialized
    ])
