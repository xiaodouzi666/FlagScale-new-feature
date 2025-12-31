"""Heartbeat monitoring for in-process fault detection.

This module provides heartbeat-based progress monitoring inspired by
NVIDIA's RankMonitorClient heartbeats API. Supports both monitoring-only
mode and restart-triggering mode.

Key features:
- Periodic heartbeat sending
- Timeout detection with configurable thresholds
- Dynamic timeout estimation
- Support for checkpoint/initialization phases
- Optional restart triggering on timeout
"""

import logging
import os
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .exception import HeartbeatTimeoutError, RankShouldRestart
from .state import FrozenRankState, HealthStatus, RankState

logger = logging.getLogger(__name__)


class HeartbeatPhase(Enum):
    """Training phase for timeout adjustment."""

    INITIALIZATION = "initialization"  # Model/data loading
    TRAINING = "training"  # Normal training
    CHECKPOINT = "checkpoint"  # Checkpoint saving/loading
    EVALUATION = "evaluation"  # Evaluation/validation
    CUSTOM = "custom"  # User-defined phase


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat monitoring.

    Attributes:
        interval: Base interval between heartbeats in seconds
        timeout: Default timeout for detecting failures
        init_timeout: Timeout during initialization phase
        checkpoint_timeout: Timeout during checkpoint operations
        auto_estimate_timeout: Whether to dynamically estimate timeout
        timeout_multiplier: Multiplier for estimated timeout
        min_timeout: Minimum allowed timeout
        max_timeout: Maximum allowed timeout
    """

    interval: float = 10.0  # seconds
    timeout: float = 60.0  # seconds
    init_timeout: float = 300.0  # 5 minutes for initialization
    checkpoint_timeout: float = 600.0  # 10 minutes for checkpointing
    auto_estimate_timeout: bool = True
    timeout_multiplier: float = 3.0
    min_timeout: float = 30.0
    max_timeout: float = 3600.0  # 1 hour


@dataclass
class HeartbeatRecord:
    """Record of a single heartbeat.

    Attributes:
        timestamp: When the heartbeat was sent
        iteration: Training iteration at heartbeat time
        phase: Current training phase
        metrics: Additional metrics sent with heartbeat
    """

    timestamp: float
    iteration: int = 0
    phase: HeartbeatPhase = HeartbeatPhase.TRAINING
    metrics: Dict[str, Any] = field(default_factory=dict)


class HeartbeatSender:
    """Sends periodic heartbeats from a rank.

    This class runs in a separate thread and periodically sends heartbeat
    signals to indicate the rank is alive and making progress.
    """

    def __init__(
        self,
        rank: int,
        config: HeartbeatConfig = None,
        on_heartbeat: Callable[[HeartbeatRecord], None] = None,
    ):
        """Initialize heartbeat sender.

        Args:
            rank: The rank sending heartbeats
            config: Heartbeat configuration
            on_heartbeat: Callback invoked on each heartbeat
        """
        self.rank = rank
        self.config = config or HeartbeatConfig()
        self.on_heartbeat = on_heartbeat

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._current_iteration = 0
        self._current_phase = HeartbeatPhase.INITIALIZATION
        self._last_heartbeat: Optional[HeartbeatRecord] = None
        self._heartbeat_history: List[HeartbeatRecord] = []
        self._max_history = 100

    def start(self) -> None:
        """Start the heartbeat sender thread."""
        if self._running:
            logger.warning("Heartbeat sender already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"HeartbeatSender-rank{self.rank}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Heartbeat sender started for rank {self.rank} "
            f"with interval {self.config.interval}s"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the heartbeat sender thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Heartbeat sender thread did not stop in time")

        logger.info(f"Heartbeat sender stopped for rank {self.rank}")

    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop running in separate thread."""
        logger.debug(f"Heartbeat loop started for rank {self.rank}")

        while self._running and not self._stop_event.is_set():
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=self.config.interval)

        logger.debug(f"Heartbeat loop ended for rank {self.rank}")

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat."""
        with self._lock:
            record = HeartbeatRecord(
                timestamp=time.time(),
                iteration=self._current_iteration,
                phase=self._current_phase,
                metrics={
                    "rank": self.rank,
                    "pid": os.getpid(),
                },
            )
            self._last_heartbeat = record
            self._heartbeat_history.append(record)

            # Trim history
            if len(self._heartbeat_history) > self._max_history:
                self._heartbeat_history = self._heartbeat_history[-self._max_history:]

        # Invoke callback
        if self.on_heartbeat:
            try:
                self.on_heartbeat(record)
            except Exception as e:
                logger.error(f"Error in heartbeat callback: {e}")

        logger.debug(
            f"Heartbeat sent: rank={self.rank}, iter={record.iteration}, "
            f"phase={record.phase.value}"
        )

    def ping(self, iteration: int = None, phase: HeartbeatPhase = None) -> None:
        """Manually trigger a heartbeat update.

        Call this to update progress information between automatic heartbeats.

        Args:
            iteration: Current training iteration
            phase: Current training phase
        """
        with self._lock:
            if iteration is not None:
                self._current_iteration = iteration
            if phase is not None:
                self._current_phase = phase

    def set_phase(self, phase: HeartbeatPhase) -> None:
        """Set the current training phase.

        Args:
            phase: The training phase to set
        """
        with self._lock:
            self._current_phase = phase
        logger.debug(f"Heartbeat phase set to {phase.value} for rank {self.rank}")

    def get_last_heartbeat(self) -> Optional[HeartbeatRecord]:
        """Get the last heartbeat record."""
        with self._lock:
            return self._last_heartbeat

    def get_history(self) -> List[HeartbeatRecord]:
        """Get heartbeat history."""
        with self._lock:
            return self._heartbeat_history.copy()


class HeartbeatMonitor:
    """Monitors heartbeats from multiple ranks.

    This class tracks heartbeats from all ranks and detects timeouts.
    Supports both monitoring-only mode and restart-triggering mode.
    """

    def __init__(
        self,
        world_size: int,
        config: HeartbeatConfig = None,
        on_timeout: Callable[[int, float], None] = None,
        on_recovered: Callable[[int], None] = None,
        enable_restart_on_timeout: bool = False,
        on_restart_needed: Callable[[int, str], None] = None,
    ):
        """Initialize heartbeat monitor.

        Args:
            world_size: Total number of ranks to monitor
            config: Heartbeat configuration
            on_timeout: Callback when timeout is detected (rank, last_heartbeat_age)
            on_recovered: Callback when rank recovers from timeout
            enable_restart_on_timeout: If True, raise RankShouldRestart on timeout
            on_restart_needed: Callback when restart is needed (rank, reason)
        """
        self.world_size = world_size
        self.config = config or HeartbeatConfig()
        self.on_timeout = on_timeout
        self.on_recovered = on_recovered
        self.enable_restart_on_timeout = enable_restart_on_timeout
        self.on_restart_needed = on_restart_needed

        self._lock = threading.Lock()
        self._last_heartbeats: Dict[int, HeartbeatRecord] = {}
        self._timeout_status: Dict[int, bool] = {}  # True if currently timed out
        self._estimated_intervals: Dict[int, float] = {}

        # Statistics
        self._timeout_counts: Dict[int, int] = {}
        self._recovery_counts: Dict[int, int] = {}

    def record_heartbeat(self, rank: int, record: HeartbeatRecord) -> None:
        """Record a heartbeat from a rank.

        Args:
            rank: The rank that sent the heartbeat
            record: The heartbeat record
        """
        with self._lock:
            prev_record = self._last_heartbeats.get(rank)
            self._last_heartbeats[rank] = record

            # Update interval estimation
            if prev_record and self.config.auto_estimate_timeout:
                interval = record.timestamp - prev_record.timestamp
                self._update_interval_estimate(rank, interval)

            # Check for recovery from timeout
            if self._timeout_status.get(rank, False):
                self._timeout_status[rank] = False
                self._recovery_counts[rank] = self._recovery_counts.get(rank, 0) + 1
                logger.info(f"Rank {rank} recovered from timeout")

                if self.on_recovered:
                    try:
                        self.on_recovered(rank)
                    except Exception as e:
                        logger.error(f"Error in recovery callback: {e}")

    def _update_interval_estimate(self, rank: int, interval: float) -> None:
        """Update the estimated heartbeat interval for a rank.

        Uses exponential moving average for smooth estimation.
        """
        alpha = 0.3  # Smoothing factor
        prev_estimate = self._estimated_intervals.get(rank, interval)
        new_estimate = alpha * interval + (1 - alpha) * prev_estimate
        self._estimated_intervals[rank] = new_estimate

    def get_effective_timeout(self, rank: int) -> float:
        """Get the effective timeout for a rank.

        Considers the current phase and dynamic estimation.
        """
        record = self._last_heartbeats.get(rank)
        base_timeout = self.config.timeout

        # Adjust for phase
        if record:
            if record.phase == HeartbeatPhase.INITIALIZATION:
                base_timeout = self.config.init_timeout
            elif record.phase == HeartbeatPhase.CHECKPOINT:
                base_timeout = self.config.checkpoint_timeout

        # Apply dynamic estimation
        if self.config.auto_estimate_timeout and rank in self._estimated_intervals:
            estimated = self._estimated_intervals[rank] * self.config.timeout_multiplier
            # Use max of base and estimated, clamped to limits
            dynamic_timeout = max(base_timeout, estimated)
            dynamic_timeout = max(self.config.min_timeout, dynamic_timeout)
            dynamic_timeout = min(self.config.max_timeout, dynamic_timeout)
            return dynamic_timeout

        return base_timeout

    def check_timeouts(self) -> List[int]:
        """Check all ranks for heartbeat timeouts.

        Returns:
            List of ranks that have timed out

        Raises:
            RankShouldRestart: If enable_restart_on_timeout is True and timeout detected
        """
        current_time = time.time()
        timed_out_ranks = []
        restart_triggered = False
        restart_reason = None

        with self._lock:
            for rank in range(self.world_size):
                record = self._last_heartbeats.get(rank)

                if record is None:
                    # No heartbeat received yet
                    continue

                age = current_time - record.timestamp
                timeout = self.get_effective_timeout(rank)

                if age > timeout:
                    if not self._timeout_status.get(rank, False):
                        # New timeout detected
                        self._timeout_status[rank] = True
                        self._timeout_counts[rank] = (
                            self._timeout_counts.get(rank, 0) + 1
                        )

                        logger.warning(
                            f"Heartbeat timeout for rank {rank}: "
                            f"last heartbeat {age:.1f}s ago (timeout: {timeout:.1f}s)"
                        )

                        if self.on_timeout:
                            try:
                                self.on_timeout(rank, age)
                            except Exception as e:
                                logger.error(f"Error in timeout callback: {e}")

                        # Check if we should trigger restart
                        if self.enable_restart_on_timeout:
                            restart_triggered = True
                            restart_reason = (
                                f"Heartbeat timeout for rank {rank}: "
                                f"last heartbeat {age:.1f}s ago"
                            )

                            if self.on_restart_needed:
                                try:
                                    self.on_restart_needed(rank, restart_reason)
                                except Exception as e:
                                    logger.error(f"Error in restart callback: {e}")

                    timed_out_ranks.append(rank)

        # Raise restart exception after releasing lock
        if restart_triggered and restart_reason:
            raise RankShouldRestart(
                reason=restart_reason,
                rank=timed_out_ranks[0] if timed_out_ranks else None,
                fault_type="heartbeat_timeout",
            )

        return timed_out_ranks

    def trigger_restart_for_rank(self, rank: int, reason: str = None) -> None:
        """Manually trigger a restart for a specific rank.

        Args:
            rank: The rank to restart
            reason: Reason for the restart

        Raises:
            RankShouldRestart: Always raised to trigger the restart
        """
        restart_reason = reason or f"Manual restart triggered for rank {rank}"

        if self.on_restart_needed:
            try:
                self.on_restart_needed(rank, restart_reason)
            except Exception as e:
                logger.error(f"Error in restart callback: {e}")

        raise RankShouldRestart(
            reason=restart_reason,
            rank=rank,
            fault_type="manual",
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        with self._lock:
            current_time = time.time()
            ranks_status = {}

            for rank in range(self.world_size):
                record = self._last_heartbeats.get(rank)
                if record:
                    age = current_time - record.timestamp
                    timeout = self.get_effective_timeout(rank)
                    ranks_status[rank] = {
                        "last_heartbeat": record.timestamp,
                        "age_seconds": age,
                        "timeout": timeout,
                        "phase": record.phase.value,
                        "iteration": record.iteration,
                        "timed_out": self._timeout_status.get(rank, False),
                        "timeout_count": self._timeout_counts.get(rank, 0),
                        "recovery_count": self._recovery_counts.get(rank, 0),
                    }
                else:
                    ranks_status[rank] = {
                        "last_heartbeat": None,
                        "status": "no_heartbeat_received",
                    }

            return {
                "world_size": self.world_size,
                "ranks": ranks_status,
                "total_timeouts": sum(self._timeout_counts.values()),
                "currently_timed_out": sum(
                    1 for v in self._timeout_status.values() if v
                ),
            }

    def get_healthy_ranks(self) -> List[int]:
        """Get list of ranks that are not timed out."""
        with self._lock:
            return [
                rank
                for rank in range(self.world_size)
                if not self._timeout_status.get(rank, False)
                and rank in self._last_heartbeats
            ]

    def get_timed_out_ranks(self) -> List[int]:
        """Get list of ranks that are currently timed out."""
        with self._lock:
            return [
                rank
                for rank, timed_out in self._timeout_status.items()
                if timed_out
            ]


class RankMonitorClient:
    """Client for rank-level monitoring.

    This class provides a simple API for training code to integrate
    heartbeat monitoring, inspired by NVIDIA's RankMonitorClient.

    Example usage:
        client = RankMonitorClient.init_workload_monitoring(
            rank=rank,
            world_size=world_size,
        )

        for batch in dataloader:
            # Training step
            train_step(batch)

            # Send heartbeat
            client.send_heartbeat(iteration=step)
    """

    _instance: Optional["RankMonitorClient"] = None

    def __init__(
        self,
        rank: int,
        world_size: int,
        config: HeartbeatConfig = None,
    ):
        """Initialize rank monitor client.

        Args:
            rank: Local rank
            world_size: Total number of ranks
            config: Heartbeat configuration
        """
        self.rank = rank
        self.world_size = world_size
        self.config = config or HeartbeatConfig()

        self._sender = HeartbeatSender(rank, config)
        self._started = False

    @classmethod
    def init_workload_monitoring(
        cls,
        rank: int = None,
        world_size: int = None,
        config: HeartbeatConfig = None,
    ) -> "RankMonitorClient":
        """Initialize workload monitoring (singleton pattern).

        Args:
            rank: Local rank. If None, reads from RANK env var.
            world_size: Total ranks. If None, reads from WORLD_SIZE env var.
            config: Heartbeat configuration

        Returns:
            RankMonitorClient instance
        """
        if cls._instance is not None:
            logger.warning("RankMonitorClient already initialized")
            return cls._instance

        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        cls._instance = cls(rank, world_size, config)
        cls._instance.start()

        logger.info(
            f"Workload monitoring initialized for rank {rank}/{world_size}"
        )
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional["RankMonitorClient"]:
        """Get the singleton instance."""
        return cls._instance

    def start(self) -> None:
        """Start heartbeat sending."""
        if self._started:
            return
        self._sender.start()
        self._started = True

    def stop(self) -> None:
        """Stop heartbeat sending."""
        if not self._started:
            return
        self._sender.stop()
        self._started = False

    def send_heartbeat(
        self,
        iteration: int = None,
        phase: HeartbeatPhase = None,
    ) -> None:
        """Send a heartbeat signal.

        Args:
            iteration: Current training iteration
            phase: Current training phase
        """
        self._sender.ping(iteration, phase)

    def enter_checkpoint_phase(self) -> None:
        """Signal entering checkpoint save/load phase."""
        self._sender.set_phase(HeartbeatPhase.CHECKPOINT)
        logger.debug("Entered checkpoint phase")

    def exit_checkpoint_phase(self) -> None:
        """Signal exiting checkpoint phase."""
        self._sender.set_phase(HeartbeatPhase.TRAINING)
        logger.debug("Exited checkpoint phase")

    def enter_initialization_phase(self) -> None:
        """Signal entering initialization phase."""
        self._sender.set_phase(HeartbeatPhase.INITIALIZATION)
        logger.debug("Entered initialization phase")

    def exit_initialization_phase(self) -> None:
        """Signal exiting initialization phase."""
        self._sender.set_phase(HeartbeatPhase.TRAINING)
        logger.debug("Exited initialization phase")

    def __enter__(self) -> "RankMonitorClient":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def calculate_and_set_hb_timeouts(
    config: HeartbeatConfig,
    observed_intervals: List[float],
    multiplier: float = 3.0,
) -> HeartbeatConfig:
    """Calculate and update heartbeat timeouts based on observed intervals.

    This function analyzes observed heartbeat intervals and sets appropriate
    timeout values, accounting for variance in intervals.

    Args:
        config: Current heartbeat configuration
        observed_intervals: List of observed heartbeat intervals in seconds
        multiplier: Multiplier for setting timeout from max interval

    Returns:
        Updated HeartbeatConfig with calculated timeouts
    """
    if not observed_intervals:
        return config

    import statistics

    mean_interval = statistics.mean(observed_intervals)
    max_interval = max(observed_intervals)

    if len(observed_intervals) > 1:
        stdev = statistics.stdev(observed_intervals)
    else:
        stdev = 0

    # Calculate timeout as max(max_interval, mean + 3*stdev) * multiplier
    calculated_timeout = max(max_interval, mean_interval + 3 * stdev) * multiplier

    # Clamp to limits
    calculated_timeout = max(config.min_timeout, calculated_timeout)
    calculated_timeout = min(config.max_timeout, calculated_timeout)

    logger.info(
        f"Calculated heartbeat timeout: {calculated_timeout:.1f}s "
        f"(mean: {mean_interval:.1f}s, max: {max_interval:.1f}s, stdev: {stdev:.1f}s)"
    )

    config.timeout = calculated_timeout
    return config
