"""Wrapper for in-process monitoring of training functions.

This module provides a Wrapper class that wraps training functions to
automatically enable heartbeat monitoring and health checks. Inspired by
NVIDIA's nvidia-resiliency-ext Wrapper and AWS's HPWrapper.

Current Status: Monitoring-only (no fault handling/restart logic)

Example Usage:
    # As decorator
    @Wrapper(heartbeat_interval=10.0)
    def train():
        for step in range(1000):
            train_step()

    # As context manager
    with Wrapper() as wrapper:
        train()

    # Wrap existing function
    wrapper = Wrapper()
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

from .exception import HealthCheckError, MonitorError
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

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "WrapperConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})


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
        # Callbacks
        on_health_check_failed: Callable[[HealthCheckResult], None] = None,
        on_fault: Callable[[str, Optional[Exception]], None] = None,
        on_iteration: Callable[[int], None] = None,
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
            on_health_check_failed: Callback when health check fails
            on_fault: Callback when fault is recorded
            on_iteration: Callback on each iteration ping
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

        # Callbacks
        self.on_health_check_failed = on_health_check_failed
        self.on_fault = on_fault
        self.on_iteration = on_iteration

        # Initialize from environment
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # State
        self._monitor: Optional[InProcessMonitor] = None
        self._started = False
        self._iteration = 0

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
        self._monitor = InProcessMonitor(
            rank=self.rank,
            world_size=self.world_size,
            heartbeat_config=self._build_heartbeat_config(),
            health_checks=self._build_health_checks(),
            health_check_interval=self.config.health_check_interval,
            event_callback=self._on_event,
            log_dir=self.config.log_dir,
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

        Args:
            iteration: Current training iteration
            phase: Current training phase
            metrics: Additional metrics to record
        """
        if not self._started or not self._monitor:
            return

        if iteration is not None:
            self._iteration = iteration

        self._monitor.ping(iteration, phase, metrics)

        if self.on_iteration and iteration is not None:
            self.on_iteration(iteration)

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

        Args:
            fn: Function to run
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Return value of the function
        """
        self.start()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.record_fault(str(e), e)
            raise
        finally:
            self.stop()

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
