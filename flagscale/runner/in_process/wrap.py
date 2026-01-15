"""Wrapper for in-process monitoring of training functions with restart support.

This module provides a Wrapper class that wraps training functions to
automatically enable heartbeat monitoring, health checks, and automatic
restart on fault detection. Inspired by NVIDIA's nvidia-resiliency-ext
Wrapper and AWS's HPWrapper.

Features:
- Heartbeat monitoring for detecting hung processes
- Health checks (CUDA, NVML, network)
- Automatic restart on fault detection
- Configurable retry limits and backoff

Example Usage:
    # As decorator
    @Wrapper(heartbeat_interval=10.0)
    def train():
        for step in range(1000):
            train_step()

    # As context manager
    with Wrapper() as wrapper:
        train()

    # Wrap existing function with restart support
    wrapper = Wrapper(enable_restart=True, max_restarts=3)
    wrapper.run(train_fn, args, kwargs)
"""

import functools
import logging
import os
import signal
import sys
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .abort import Abort, ComposedAbort, AbortTorchDistributed, AbortCUDA, create_default_abort_handler
from .exception import HealthCheckError, MonitorError, RankShouldRestart, RestartAbort
from .restart_sync import RestartCoordinator, create_restart_coordinator
from .initialize import RetryController, RestartConfig, create_default_retry_controller
from .health_check import (
    ChainedHealthCheck,
    CudaHealthCheck,
    FaultCounter,
    HealthCheck,
    HealthCheckResult,
    HealthCheckRunner,
    NetworkHealthCheck,
    NvmlHealthCheck,
)
from .heartbeat import (
    HeartbeatConfig,
    HeartbeatMonitor,
    HeartbeatPhase,
    HeartbeatRecord,
    HeartbeatSender,
    RankMonitorClient,
)
from .monitor import InProcessMonitor, MonitorEvent, MonitorEventRecord
from .progress_watchdog import ProgressWatchdog
from .state import FrozenRankState, HealthStatus, RankMode, RankState

logger = logging.getLogger(__name__)


@dataclass
class WrapperConfig:
    """Configuration for the Wrapper.

    Attributes:
        heartbeat_interval: Interval between heartbeats in seconds
        heartbeat_timeout: Timeout for heartbeat detection
        health_check_interval: Interval between health checks
        enable_cuda_health_check: Enable CUDA/GPU health checking
        enable_nvml_health_check: Enable NVML-based GPU monitoring
        enable_network_health_check: Enable network interface checking
        max_rank_faults: Maximum faults before warning
        log_dir: Directory for monitoring logs
        init_timeout: Timeout during initialization phase
        checkpoint_timeout: Timeout during checkpoint operations

        # Restart-related configuration
        enable_restart: Enable automatic restart on fault detection
        max_restarts: Maximum number of restart attempts (0 = unlimited)
        min_world_size: Minimum world size to continue restarting
        restart_on_health_check_fail: Restart when health check fails
        restart_on_heartbeat_timeout: Restart when heartbeat times out
        restart_on_hang: Restart when hang is detected
        restart_on_exception: Restart on uncaught exceptions
        restart_delay: Base delay between restart attempts (seconds)
        exponential_backoff: Use exponential backoff for restart delays
        max_restart_delay: Maximum delay between restarts (seconds)
    """

    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 60.0
    health_check_interval: float = 60.0
    enable_cuda_health_check: bool = True
    enable_nvml_health_check: bool = True
    enable_network_health_check: bool = False
    max_rank_faults: int = 5
    log_dir: Optional[str] = None
    init_timeout: float = 300.0
    checkpoint_timeout: float = 600.0

    # Restart-related configuration
    enable_restart: bool = False
    max_restarts: int = 3
    min_world_size: int = 1
    restart_on_health_check_fail: bool = True
    restart_on_heartbeat_timeout: bool = True
    restart_on_hang: bool = True
    restart_on_exception: bool = True
    restart_delay: float = 1.0
    exponential_backoff: bool = True
    max_restart_delay: float = 60.0

    # Cross-rank synchronization for restart
    # Use a longer timeout for multi-node scenarios and slow cleanup operations
    restart_sync_barrier_timeout: float = 300.0  # Timeout for restart sync barriers

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "WrapperConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})

    def get_restart_delay(self, restart_attempt: int) -> float:
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


class Wrapper:
    """Wrapper for training functions with automatic monitoring.

    This wrapper automatically starts heartbeat monitoring and health checks
    when wrapping a training function. It can be used as a decorator, context
    manager, or by calling the run() method directly.

    In monitoring-only mode, faults are logged but do not trigger restarts.

    Example:
        # As decorator
        @Wrapper(heartbeat_interval=10.0)
        def train():
            for step in range(1000):
                train_step()

        # As context manager
        with Wrapper() as wrapper:
            train()
            wrapper.ping(iteration=step)

        # Direct usage
        wrapper = Wrapper()
        result = wrapper.run(train_fn)
    """

    # Singleton instance for global access
    _instance: Optional["Wrapper"] = None

    def __init__(
        self,
        config: Union[WrapperConfig, Dict[str, Any]] = None,
        # Individual config options (override config)
        heartbeat_interval: float = None,
        heartbeat_timeout: float = None,
        health_check_interval: float = None,
        enable_cuda_health_check: bool = None,
        enable_nvml_health_check: bool = None,
        enable_network_health_check: bool = None,
        max_rank_faults: int = None,
        log_dir: str = None,
        # Restart-related options
        enable_restart: bool = None,
        max_restarts: int = None,
        min_world_size: int = None,
        restart_on_exception: bool = None,
        # Callbacks
        on_health_check_failed: Callable[[HealthCheckResult], None] = None,
        on_fault: Callable[[str, Optional[Exception]], None] = None,
        on_iteration: Callable[[int], None] = None,
        on_restart: Callable[[int, str], None] = None,
    ):
        """Initialize the Wrapper.

        Args:
            config: WrapperConfig or dict with configuration
            heartbeat_interval: Override heartbeat interval
            heartbeat_timeout: Override heartbeat timeout
            health_check_interval: Override health check interval
            enable_cuda_health_check: Override CUDA health check
            enable_nvml_health_check: Override NVML health check
            enable_network_health_check: Override network health check
            max_rank_faults: Override max faults
            log_dir: Override log directory
            enable_restart: Enable automatic restart on fault
            max_restarts: Maximum restart attempts
            min_world_size: Minimum world size to continue
            restart_on_exception: Restart on uncaught exceptions
            on_health_check_failed: Callback when health check fails
            on_fault: Callback when fault is recorded
            on_iteration: Callback on each iteration ping
            on_restart: Callback when restart is triggered (iteration, reason)
        """
        # Build config
        if isinstance(config, dict):
            self.config = WrapperConfig.from_dict(config)
        elif config is not None:
            self.config = config
        else:
            self.config = WrapperConfig()

        # Apply overrides
        if heartbeat_interval is not None:
            self.config.heartbeat_interval = heartbeat_interval
        if heartbeat_timeout is not None:
            self.config.heartbeat_timeout = heartbeat_timeout
        if health_check_interval is not None:
            self.config.health_check_interval = health_check_interval
        if enable_cuda_health_check is not None:
            self.config.enable_cuda_health_check = enable_cuda_health_check
        if enable_nvml_health_check is not None:
            self.config.enable_nvml_health_check = enable_nvml_health_check
        if enable_network_health_check is not None:
            self.config.enable_network_health_check = enable_network_health_check
        if max_rank_faults is not None:
            self.config.max_rank_faults = max_rank_faults
        if log_dir is not None:
            self.config.log_dir = log_dir
        if enable_restart is not None:
            self.config.enable_restart = enable_restart
        if max_restarts is not None:
            self.config.max_restarts = max_restarts
        if min_world_size is not None:
            self.config.min_world_size = min_world_size
        if restart_on_exception is not None:
            self.config.restart_on_exception = restart_on_exception

        # Callbacks
        self.on_health_check_failed = on_health_check_failed
        self.on_fault = on_fault
        self.on_iteration = on_iteration
        self.on_restart = on_restart

        # Initialize from environment
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Restart-related state
        self._state = RankState.from_env()
        self._retry_controller: Optional[RetryController] = None
        self._abort_handler: Optional[Abort] = None
        self._restart_coordinator: Optional[RestartCoordinator] = None

        # Initialize restart components if enabled
        if self.config.enable_restart:
            self._retry_controller = create_default_retry_controller(
                max_restarts=self.config.max_restarts,
                min_world_size=self.config.min_world_size,
            )
            self._abort_handler = create_default_abort_handler()

            # Initialize RestartCoordinator for cross-rank synchronization
            self._restart_coordinator = create_restart_coordinator(
                rank=self.rank,
                world_size=self.world_size,
                timeout=self.config.restart_sync_barrier_timeout,
            )

            # Rank 0 eagerly initializes the TCPStore server
            # This ensures the server is ready before other ranks try to connect
            if self.rank == 0 and self._restart_coordinator is not None:
                logger.info("Rank 0: Eagerly initializing RestartCoordinator TCPStore...")
                self._restart_coordinator._ensure_store()

        # State
        self._monitor: Optional[InProcessMonitor] = None
        self._started = False
        self._iteration = 0
        self._in_restart_loop = False  # True when running via run_with_restart()

        # Set singleton
        Wrapper._instance = self

    @classmethod
    def get_instance(cls) -> Optional["Wrapper"]:
        """Get the singleton Wrapper instance."""
        return cls._instance

    def _build_health_checks(self) -> List[HealthCheck]:
        """Build list of health checks based on config."""
        checks = []

        if self.config.enable_cuda_health_check:
            checks.append(CudaHealthCheck())

        if self.config.enable_nvml_health_check:
            checks.append(NvmlHealthCheck())

        if self.config.enable_network_health_check:
            checks.append(NetworkHealthCheck())

        checks.append(FaultCounter(max_rank_faults=self.config.max_rank_faults))

        return checks

    def _build_heartbeat_config(self) -> HeartbeatConfig:
        """Build heartbeat config."""
        return HeartbeatConfig(
            interval=self.config.heartbeat_interval,
            timeout=self.config.heartbeat_timeout,
            init_timeout=self.config.init_timeout,
            checkpoint_timeout=self.config.checkpoint_timeout,
        )

    def _on_event(self, event: MonitorEventRecord) -> None:
        """Handle monitoring events."""
        if event.event_type == MonitorEvent.HEALTH_CHECK:
            if not event.data.get("healthy", True):
                if self.on_health_check_failed:
                    result = HealthCheckResult(
                        healthy=False,
                        check_name=event.data.get("check_name", "unknown"),
                        reason=event.data.get("reason", ""),
                        metrics=event.data.get("metrics", {}),
                    )
                    self.on_health_check_failed(result)

        elif event.event_type == MonitorEvent.FAULT:
            if self.on_fault:
                self.on_fault(
                    event.data.get("reason", ""),
                    event.data.get("error"),
                )

    def start(self) -> "Wrapper":
        """Start the wrapper monitoring.

        Returns:
            Self for chaining
        """
        if self._started:
            logger.warning("Wrapper already started")
            return self

        # Create and start monitor
        # Enable restart triggering on health check failure if restart is enabled
        enable_restart_on_failure = (
            self.config.enable_restart and
            self.config.restart_on_health_check_fail
        )

        self._monitor = InProcessMonitor(
            rank=self.rank,
            world_size=self.world_size,
            heartbeat_config=self._build_heartbeat_config(),
            health_checks=self._build_health_checks(),
            health_check_interval=self.config.health_check_interval,
            event_callback=self._on_event,
            log_dir=self.config.log_dir,
            enable_restart_on_failure=enable_restart_on_failure,
        )
        self._monitor.start()
        self._started = True

        logger.info(
            f"Wrapper started for rank {self.rank}/{self.world_size} "
            f"(heartbeat: {self.config.heartbeat_interval}s, "
            f"health_check: {self.config.health_check_interval}s)"
        )

        return self

    def stop(self) -> None:
        """Stop the wrapper monitoring."""
        if not self._started:
            return

        if self._monitor:
            self._monitor.stop()
            self._monitor = None

        self._started = False
        logger.info(f"Wrapper stopped for rank {self.rank}")

    def ping(
        self,
        iteration: int = None,
        phase: HeartbeatPhase = None,
        metrics: Dict[str, Any] = None,
    ) -> None:
        """Send a heartbeat ping.

        Call this periodically in training to update progress.
        Also checks if any peer rank has requested restart (passive sync).

        Args:
            iteration: Current training iteration
            phase: Current training phase
            metrics: Additional metrics to record

        Raises:
            RankShouldRestart: If a peer rank has requested restart
        """
        if not self._started or not self._monitor:
            return

        # Manual restart trigger via file (opt-in)
        # Only rank 0 checks the trigger file; other ranks will follow via RestartCoordinator
        trigger_file = os.environ.get("FLAGSCALE_RESTART_TRIGGER_FILE", "")
        if trigger_file and self.rank == 0:
            try:
                if os.path.exists(trigger_file):
                    logger.warning(
                        f"Rank {self.rank}: Manual restart trigger file detected: {trigger_file}"
                    )
                    # Best-effort remove to make it one-shot
                    try:
                        os.remove(trigger_file)
                    except Exception:
                        pass
                    # Only trigger restart if we're in a restart loop context
                    if self._in_restart_loop:
                        # Use the unified path: broadcast + raise RankShouldRestart
                        self.trigger_restart(reason=f"Manual file trigger: {trigger_file}")
                    else:
                        logger.warning(
                            f"Rank {self.rank}: Restart trigger detected but not in restart loop context. "
                            f"To enable restart, use wrapper.run_with_restart() or run_with_restart=True."
                        )
            except RankShouldRestart:
                # Re-raise RankShouldRestart so it propagates to the restart handler
                raise
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Manual trigger check failed: {e}")

        if iteration is not None:
            self._iteration = iteration

        self._monitor.ping(iteration, phase, metrics)

        if self.on_iteration and iteration is not None:
            self.on_iteration(iteration)

        # Check if any peer rank has requested restart (passive sync)
        # Only check and raise if we're in a restart loop context
        if self._in_restart_loop and self._restart_coordinator is not None:
            if self._restart_coordinator.restart_requested(self._state.restart_attempt):
                reason = self._restart_coordinator.get_reason(self._state.restart_attempt)
                logger.info(
                    f"Rank {self.rank}: Detected peer restart request (attempt {self._state.restart_attempt}), "
                    f"reason: {reason}"
                )
                raise RankShouldRestart(
                    reason=f"Peer restart: {reason}",
                    rank=self.rank,
                    fault_type="peer_restart",
                )

    def enter_checkpoint_phase(self) -> None:
        """Signal entering checkpoint save/load phase."""
        if self._monitor:
            self._monitor.enter_checkpoint_phase()

    def exit_checkpoint_phase(self) -> None:
        """Signal exiting checkpoint phase."""
        if self._monitor:
            self._monitor.exit_checkpoint_phase()

    def enter_initialization_phase(self) -> None:
        """Signal entering initialization phase."""
        if self._monitor:
            self._monitor.enter_initialization_phase()

    def exit_initialization_phase(self) -> None:
        """Signal exiting initialization phase."""
        if self._monitor:
            self._monitor.exit_initialization_phase()

    def record_fault(self, reason: str, error: Exception = None) -> int:
        """Record a fault occurrence.

        Args:
            reason: Description of the fault
            error: Associated exception if any

        Returns:
            Current fault count
        """
        if self._monitor:
            return self._monitor.record_fault(reason, error)
        return 0

    def get_state(self) -> Optional[FrozenRankState]:
        """Get current frozen state."""
        if self._monitor:
            return self._monitor.get_state()
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if self._monitor:
            return self._monitor.get_status()
        return {"started": False}

    def run(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a function with monitoring.

        If enable_restart is True, uses run_with_restart for automatic
        fault recovery. Otherwise, runs the function once with monitoring.

        Args:
            fn: Function to run
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Return value of the function
        """
        if self.config.enable_restart:
            return self.run_with_restart(fn, *args, **kwargs)

        self.start()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.record_fault(str(e), e)
            raise
        finally:
            self.stop()

    def run_with_restart(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a function with automatic restart on fault.

        This method implements the core restart loop. When a fault is
        detected (through health checks, heartbeat timeout, or exception),
        it will abort, clean up resources, and restart the function.

        Args:
            fn: Function to run
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Return value of the function

        Raises:
            RestartAbort: When max restarts exceeded or world size too small
        """
        if not self._retry_controller:
            self._retry_controller = create_default_retry_controller(
                max_restarts=self.config.max_restarts,
                min_world_size=self.config.min_world_size,
            )

        if not self._abort_handler:
            self._abort_handler = create_default_abort_handler()

        result = None
        last_error = None
        self._in_restart_loop = True  # Mark that we're in the restart loop context

        while True:
            # Get frozen state for checks
            frozen_state = FrozenRankState.from_state(self._state)

            try:
                # 1. Check if we should continue (retry controller)
                if not self._retry_controller.should_continue(frozen_state):
                    raise RestartAbort(
                        reason=f"Retry limit reached after {self._state.restart_attempt} attempts",
                        restart_count=self._state.restart_attempt,
                    )

                # Log restart attempt
                if self._state.restart_attempt > 0:
                    logger.info(
                        f"Rank {self.rank}: Restart attempt {self._state.restart_attempt}"
                        f"/{self.config.max_restarts if self.config.max_restarts > 0 else '∞'}"
                    )
                    if self.on_restart:
                        self.on_restart(
                            self._state.restart_attempt,
                            self._state.last_restart_reason or "unknown",
                        )

                    # Apply restart delay with exponential backoff
                    delay = self.config.get_restart_delay(self._state.restart_attempt - 1)
                    if delay > 0:
                        logger.debug(f"Waiting {delay:.1f}s before restart...")
                        time.sleep(delay)

                # 2. Start monitoring
                self.start()

                # 3. Execute the training function
                self._state.set_mode(RankMode.ACTIVE)
                result = fn(*args, **kwargs)

                # 4. Success - exit the loop
                self._state.set_mode(RankMode.TERMINATED)
                logger.info(f"Rank {self.rank}: Function completed successfully")
                break

            except RankShouldRestart as e:
                # 5. Restart triggered by fault detection
                logger.warning(f"Rank {self.rank}: Restart triggered: {e.reason}")
                last_error = e.original_error

                # Broadcast restart request to peers (if not already from peer)
                if self._restart_coordinator is not None and e.fault_type != "peer_restart":
                    self._restart_coordinator.request_restart(
                        attempt=self._state.restart_attempt,
                        rank=self.rank,
                        reason=e.reason,
                        iteration=self._iteration,
                    )

                # Note: Synchronization barriers are handled in _handle_restart()
                # to avoid duplicate barrier points
                self._handle_restart(e.reason, e.original_error)
                continue

            except RestartAbort as e:
                # 6. Restart aborted - exit with error
                logger.error(f"Rank {self.rank}: Restart aborted: {e.reason}")
                raise

            except Exception as e:
                # 7. Unexpected exception
                if self.config.restart_on_exception:
                    logger.warning(
                        f"Rank {self.rank}: Exception caught, triggering restart: {e}"
                    )
                    last_error = e

                    # Broadcast restart request to peers
                    if self._restart_coordinator is not None:
                        self._restart_coordinator.request_restart(
                            attempt=self._state.restart_attempt,
                            rank=self.rank,
                            reason=f"Exception: {type(e).__name__}",
                            iteration=self._iteration,
                        )

                    # Note: Synchronization barriers are handled in _handle_restart()
                    # to avoid duplicate barrier points
                    self._handle_restart(f"Exception: {type(e).__name__}", e)
                    continue
                else:
                    # Don't restart on exceptions, just record and re-raise
                    self.record_fault(str(e), e)
                    raise

            finally:
                # Always stop monitoring
                self.stop()

        return result

    def _handle_restart(self, reason: str, error: Exception = None) -> None:
        """Handle a restart event.

        This method performs cleanup and prepares for the next restart attempt.
        Uses barriers to synchronize all ranks during restart.

        Args:
            reason: Reason for the restart
            error: The exception that caused the restart, if any
        """
        # Record the fault
        self.record_fault(reason, error)

        # Stop monitoring
        self.stop()

        # Barrier 1: Wait for all ranks to detect fault before cleanup
        # This ensures no rank starts cleanup while others are still running
        if self._restart_coordinator is not None:
            logger.info(f"Rank {self.rank}: Waiting at fault_detected barrier...")
            barrier_ok = self._restart_coordinator.barrier(
                name="fault_detected",
                attempt=self._state.restart_attempt,
                timeout_s=self.config.restart_sync_barrier_timeout,
            )
            if not barrier_ok:
                logger.warning(
                    f"Rank {self.rank}: fault_detected barrier timeout, proceeding anyway"
                )

        # Get frozen state for abort handlers
        frozen_state = FrozenRankState.from_state(self._state)

        # Execute abort handlers (cleanup)
        if self._abort_handler:
            try:
                self._abort_handler(frozen_state)
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Abort handler error: {e}")

        # Barrier 2: Wait for all ranks to complete cleanup before restarting
        # This ensures no rank starts reinitializing while others are still cleaning up
        if self._restart_coordinator is not None:
            logger.info(f"Rank {self.rank}: Waiting at abort_done barrier...")
            barrier_ok = self._restart_coordinator.barrier(
                name="abort_done",
                attempt=self._state.restart_attempt,
                timeout_s=self.config.restart_sync_barrier_timeout,
            )
            if not barrier_ok:
                logger.warning(
                    f"Rank {self.rank}: abort_done barrier timeout, proceeding anyway"
                )

        # Advance to next restart attempt
        self._state.advance(reason)

        logger.info(
            f"Rank {self.rank}: Prepared for restart attempt {self._state.restart_attempt}"
        )

    def trigger_restart(self, reason: str, error: Exception = None) -> None:
        """Manually trigger a restart.

        This can be called from within the training function to trigger
        a restart based on application-specific conditions.

        Args:
            reason: Reason for the restart
            error: Associated error if any

        Raises:
            RankShouldRestart: Always raised to trigger the restart
        """
        # Broadcast restart request to peers before raising exception
        if self._restart_coordinator is not None:
            self._restart_coordinator.request_restart(
                attempt=self._state.restart_attempt,
                rank=self.rank,
                reason=reason,
                iteration=self._iteration,
            )
            logger.info(
                f"Rank {self.rank}: Broadcast manual restart request: {reason}"
            )

        raise RankShouldRestart(
            reason=reason,
            rank=self.rank,
            original_error=error,
            fault_type="manual",
        )

    def __call__(self, fn: Callable) -> Callable:
        """Use as decorator.

        Args:
            fn: Function to wrap

        Returns:
            Wrapped function
        """

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return self.run(fn, *args, **kwargs)

        return wrapped

    def __enter__(self) -> "Wrapper":
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_val:
            self.record_fault(str(exc_val), exc_val)
        self.stop()


class CallWrapper:
    """Encapsulates state and execution flow for a single wrapped call.

    This class manages the lifecycle of a single invocation of a wrapped
    function, including monitoring setup and teardown.

    Note: In monitoring-only mode, this does not implement restart logic.
    """

    def __init__(
        self,
        wrapper: Wrapper,
        fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
    ):
        """Initialize the call wrapper.

        Args:
            wrapper: Parent Wrapper instance
            fn: Function to call
            args: Positional arguments
            kwargs: Keyword arguments
        """
        self.wrapper = wrapper
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}

        self.state = RankState.from_env()
        self.result = None
        self.exception = None
        self.start_time = None
        self.end_time = None

    def execute(self) -> Any:
        """Execute the wrapped function.

        Returns:
            Return value of the function
        """
        self.start_time = time.time()
        self.state.set_mode(RankMode.ACTIVE)

        try:
            self.result = self.fn(*self.args, **self.kwargs)
            return self.result

        except Exception as e:
            self.exception = e
            self.state.exception = e
            self.wrapper.record_fault(str(e), e)
            raise

        finally:
            self.end_time = time.time()
            self.state.set_mode(RankMode.TERMINATED)

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def wrap_training(
    fn: Callable = None,
    *,
    heartbeat_interval: float = 10.0,
    health_check_interval: float = 60.0,
    enable_cuda_health_check: bool = True,
    enable_nvml_health_check: bool = True,
    log_dir: str = None,
) -> Union[Callable, Wrapper]:
    """Convenience decorator for wrapping training functions.

    Can be used with or without arguments:

        @wrap_training
        def train():
            ...

        @wrap_training(heartbeat_interval=5.0)
        def train():
            ...

    Args:
        fn: Function to wrap (when used without arguments)
        heartbeat_interval: Interval between heartbeats
        health_check_interval: Interval between health checks
        enable_cuda_health_check: Enable CUDA health checking
        enable_nvml_health_check: Enable NVML monitoring
        log_dir: Directory for monitoring logs

    Returns:
        Wrapped function or decorator
    """
    wrapper = Wrapper(
        heartbeat_interval=heartbeat_interval,
        health_check_interval=health_check_interval,
        enable_cuda_health_check=enable_cuda_health_check,
        enable_nvml_health_check=enable_nvml_health_check,
        log_dir=log_dir,
    )

    if fn is not None:
        # Used as @wrap_training without arguments
        return wrapper(fn)

    # Used as @wrap_training(...) with arguments
    return wrapper


def init_in_process_monitoring(
    config: Dict[str, Any] = None,
    **kwargs,
) -> Wrapper:
    """Initialize in-process monitoring globally.

    This function creates and starts a global Wrapper instance that can
    be accessed via Wrapper.get_instance().

    Args:
        config: Configuration dictionary
        **kwargs: Additional configuration options

    Returns:
        The started Wrapper instance
    """
    if config:
        wrapper_config = WrapperConfig.from_dict(config)
    else:
        wrapper_config = WrapperConfig()

    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(wrapper_config, key):
            setattr(wrapper_config, key, value)

    wrapper = Wrapper(config=wrapper_config)
    wrapper.start()

    return wrapper


def shutdown_in_process_monitoring() -> None:
    """Shutdown the global in-process monitoring."""
    wrapper = Wrapper.get_instance()
    if wrapper:
        wrapper.stop()
