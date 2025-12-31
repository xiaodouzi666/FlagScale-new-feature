"""Distributed monitoring for in-process fault detection.

This module provides monitoring capabilities for distributed training,
including heartbeat monitoring, health checks, and metric collection.
Supports both monitoring-only mode and restart-triggering mode.
"""

import json
import logging
import os
import socket
import threading
import time

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .exception import MonitorError, StoreError, RankShouldRestart
from .health_check import (
    ChainedHealthCheck,
    CudaHealthCheck,
    FaultCounter,
    HealthCheck,
    HealthCheckResult,
    HealthCheckRunner,
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
from .state import (
    FrozenRankState,
    HealthStatus,
    RankMode,
    RankState,
    StateManager,
)

logger = logging.getLogger(__name__)


class MonitorEvent(Enum):
    """Types of monitoring events."""

    HEARTBEAT = "heartbeat"
    HEALTH_CHECK = "health_check"
    TIMEOUT = "timeout"
    RECOVERY = "recovery"
    FAULT = "fault"
    STATUS_CHANGE = "status_change"


@dataclass
class MonitorEventRecord:
    """Record of a monitoring event.

    Attributes:
        event_type: Type of event
        rank: Rank that generated the event
        timestamp: When the event occurred
        data: Additional event data
        severity: Event severity (info, warning, error)
    """

    event_type: MonitorEvent
    rank: int
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "rank": self.rank,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
            "severity": self.severity,
        }


class InProcessMonitor:
    """Main monitor class for in-process fault detection.

    This class coordinates heartbeat monitoring, health checks, and event
    logging for distributed training. It runs monitoring in a separate
    thread and provides callbacks for various events.

    Example usage:
        monitor = InProcessMonitor(
            rank=rank,
            world_size=world_size,
        )
        monitor.start()

        # In training loop
        for step, batch in enumerate(dataloader):
            train_step(batch)
            monitor.ping(iteration=step)

        monitor.stop()
    """

    def __init__(
        self,
        rank: int = None,
        world_size: int = None,
        heartbeat_config: HeartbeatConfig = None,
        health_checks: List[HealthCheck] = None,
        health_check_interval: float = 60.0,
        event_callback: Callable[[MonitorEventRecord], None] = None,
        log_dir: str = None,
        # Restart triggering options
        enable_restart_on_failure: bool = False,
        on_restart_needed: Callable[[str, Exception], None] = None,
    ):
        """Initialize the in-process monitor.

        Args:
            rank: Local rank. If None, reads from RANK env var.
            world_size: Total ranks. If None, reads from WORLD_SIZE env var.
            heartbeat_config: Configuration for heartbeat monitoring
            health_checks: List of health checks to run
            health_check_interval: Interval between health checks in seconds
            event_callback: Callback for monitoring events
            log_dir: Directory for writing monitoring logs
            enable_restart_on_failure: If True, raise RankShouldRestart on failure
            on_restart_needed: Callback when restart is needed
        """
        self.enable_restart_on_failure = enable_restart_on_failure
        self.on_restart_needed = on_restart_needed
        # Initialize rank info from environment if not provided
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.world_size = (
            world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
        )
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0)))

        # Configuration
        self.heartbeat_config = heartbeat_config or HeartbeatConfig()
        self.health_check_interval = health_check_interval
        self.event_callback = event_callback
        self.log_dir = log_dir

        # Components
        self._state = RankState(
            rank=self.rank,
            initial_rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            node_rank=self.node_rank,
            mode=RankMode.INITIALIZED,
        )

        self._heartbeat_sender = HeartbeatSender(
            rank=self.rank,
            config=self.heartbeat_config,
            on_heartbeat=self._on_local_heartbeat,
        )

        self._health_check_runner = HealthCheckRunner(
            checks=health_checks or self._default_health_checks()
        )

        # Thread management
        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None

        # Pending restart exception (from background threads)
        self._pending_restart_exception: Optional[RankShouldRestart] = None
        self._pending_exception_lock = threading.Lock()

        # Event history
        self._events: List[MonitorEventRecord] = []
        self._max_events = 1000
        self._events_lock = threading.Lock()

        # Statistics
        self._stats = {
            "start_time": None,
            "heartbeats_sent": 0,
            "health_checks_run": 0,
            "health_checks_failed": 0,
            "faults_detected": 0,
        }

        # Setup logging
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    def _default_health_checks(self) -> List[HealthCheck]:
        """Get default health checks."""
        return [
            CudaHealthCheck(),
            NvmlHealthCheck(),
            FaultCounter(max_rank_faults=5),
        ]

    def start(self) -> None:
        """Start the monitor."""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._stop_event.clear()
        self._stats["start_time"] = time.time()

        # Update state
        self._state.set_mode(RankMode.ACTIVE)
        self._state.update_heartbeat()

        # Start heartbeat sender
        self._heartbeat_sender.start()

        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name=f"HealthCheckThread-rank{self.rank}",
            daemon=True,
        )
        self._health_check_thread.start()

        logger.info(
            f"InProcessMonitor started for rank {self.rank}/{self.world_size}"
        )

        # Record start event
        self._record_event(
            MonitorEvent.STATUS_CHANGE,
            {"status": "started", "mode": RankMode.ACTIVE.value},
        )

    def stop(self) -> None:
        """Stop the monitor."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Stop heartbeat sender
        self._heartbeat_sender.stop()

        # Wait for threads
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)

        # Update state
        self._state.set_mode(RankMode.TERMINATED)

        logger.info(f"InProcessMonitor stopped for rank {self.rank}")

        # Record stop event
        self._record_event(
            MonitorEvent.STATUS_CHANGE,
            {"status": "stopped", "mode": RankMode.TERMINATED.value},
        )

        # Write final log
        self._write_final_log()

    def ping(
        self,
        iteration: int = None,
        phase: HeartbeatPhase = None,
        metrics: Dict[str, Any] = None,
    ) -> None:
        """Send a manual heartbeat ping.

        Call this periodically in training to update progress.

        Args:
            iteration: Current training iteration
            phase: Current training phase
            metrics: Additional metrics to record

        Raises:
            RankShouldRestart: If a restart was triggered by health check failure
        """
        # Check for pending restart exception from background threads
        with self._pending_exception_lock:
            if self._pending_restart_exception is not None:
                exc = self._pending_restart_exception
                self._pending_restart_exception = None
                raise exc

        if iteration is not None:
            self._state.iteration = iteration

        self._heartbeat_sender.ping(iteration, phase)

        if metrics:
            self._state.metrics.update(metrics)

    def enter_checkpoint_phase(self) -> None:
        """Signal entering checkpoint save/load phase."""
        self._heartbeat_sender.set_phase(HeartbeatPhase.CHECKPOINT)
        self._record_event(
            MonitorEvent.STATUS_CHANGE,
            {"phase": HeartbeatPhase.CHECKPOINT.value},
        )

    def exit_checkpoint_phase(self) -> None:
        """Signal exiting checkpoint phase."""
        self._heartbeat_sender.set_phase(HeartbeatPhase.TRAINING)
        self._record_event(
            MonitorEvent.STATUS_CHANGE,
            {"phase": HeartbeatPhase.TRAINING.value},
        )

    def enter_initialization_phase(self) -> None:
        """Signal entering initialization phase."""
        self._heartbeat_sender.set_phase(HeartbeatPhase.INITIALIZATION)

    def exit_initialization_phase(self) -> None:
        """Signal exiting initialization phase."""
        self._heartbeat_sender.set_phase(HeartbeatPhase.TRAINING)

    def record_fault(self, reason: str, error: Exception = None) -> int:
        """Record a fault occurrence.

        Args:
            reason: Description of the fault
            error: Associated exception if any

        Returns:
            Current fault count
        """
        fault_count = self._state.increment_fault_count()
        self._stats["faults_detected"] += 1

        self._record_event(
            MonitorEvent.FAULT,
            {
                "reason": reason,
                "error": str(error) if error else None,
                "fault_count": fault_count,
            },
            severity="warning",
        )

        logger.warning(
            f"Fault recorded for rank {self.rank}: {reason} "
            f"(total faults: {fault_count})"
        )

        return fault_count

    def trigger_restart(
        self,
        reason: str,
        error: Exception = None,
        fault_type: str = "manual",
    ) -> None:
        """Trigger a restart.

        This method raises RankShouldRestart to signal that the training
        should be restarted. Use this for application-specific fault detection.

        Args:
            reason: Reason for the restart
            error: Associated error if any
            fault_type: Type of fault (manual, health_check, heartbeat, etc.)

        Raises:
            RankShouldRestart: Always raised to trigger the restart
        """
        # Record the fault
        self.record_fault(reason, error)

        # Record restart trigger event
        self._record_event(
            MonitorEvent.FAULT,
            {
                "trigger": fault_type,
                "reason": reason,
                "restart_requested": True,
            },
            severity="error",
        )

        # Raise restart exception
        raise RankShouldRestart(
            reason=reason,
            rank=self.rank,
            original_error=error,
            fault_type=fault_type,
        )

    def get_state(self) -> FrozenRankState:
        """Get current frozen state."""
        return FrozenRankState.from_state(self._state)

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "mode": self._state.mode.value,
            "health_status": self._state.health_status.value,
            "iteration": self._state.iteration,
            "fault_count": self._state.fault_count,
            "running": self._running,
            "stats": self._stats.copy(),
            "last_heartbeat": self._heartbeat_sender.get_last_heartbeat(),
        }

    def get_events(self, limit: int = None) -> List[MonitorEventRecord]:
        """Get recorded events.

        Args:
            limit: Maximum number of events to return (most recent)

        Returns:
            List of event records
        """
        with self._events_lock:
            if limit:
                return self._events[-limit:].copy()
            return self._events.copy()

    def _on_local_heartbeat(self, record: HeartbeatRecord) -> None:
        """Handle local heartbeat event."""
        self._state.update_heartbeat()
        self._stats["heartbeats_sent"] += 1

        self._record_event(
            MonitorEvent.HEARTBEAT,
            {
                "iteration": record.iteration,
                "phase": record.phase.value,
            },
            severity="debug",
        )

    def _health_check_loop(self) -> None:
        """Background thread for running health checks."""
        logger.debug(f"Health check loop started for rank {self.rank}")

        while self._running and not self._stop_event.is_set():
            try:
                self._run_health_checks()
            except RankShouldRestart as e:
                # Store the restart exception to be raised in the main thread
                with self._pending_exception_lock:
                    self._pending_restart_exception = e
                logger.warning(f"Health check triggered restart: {e.reason}")
                # Don't break - let the main thread handle the restart
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=self.health_check_interval)

        logger.debug(f"Health check loop ended for rank {self.rank}")

    def _run_health_checks(self) -> None:
        """Run all configured health checks."""
        results = self._health_check_runner.run_all(self._state)
        self._stats["health_checks_run"] += 1

        # Update state based on results
        all_healthy = all(r.healthy for r in results)

        if all_healthy:
            self._state.set_health_status(HealthStatus.HEALTHY)
        else:
            self._state.set_health_status(HealthStatus.UNHEALTHY)
            self._stats["health_checks_failed"] += 1

            # Collect failed check info
            failed_checks = []
            for result in results:
                if not result.healthy:
                    failed_checks.append(result)
                    self._record_event(
                        MonitorEvent.HEALTH_CHECK,
                        {
                            "check_name": result.check_name,
                            "healthy": False,
                            "reason": result.reason,
                            "metrics": result.metrics,
                        },
                        severity="warning",
                    )

            # Trigger restart if enabled
            if self.enable_restart_on_failure and failed_checks:
                reason = f"Health check failed: {', '.join(r.check_name for r in failed_checks)}"
                logger.warning(f"Rank {self.rank}: {reason}")

                # Call callback if provided
                if self.on_restart_needed:
                    self.on_restart_needed(reason, None)

                # Record restart trigger event
                self._record_event(
                    MonitorEvent.FAULT,
                    {
                        "trigger": "health_check",
                        "reason": reason,
                        "failed_checks": [r.check_name for r in failed_checks],
                    },
                    severity="error",
                )

                # Raise restart exception
                raise RankShouldRestart(
                    reason=reason,
                    rank=self.rank,
                    fault_type="health_check",
                )

    def _record_event(
        self,
        event_type: MonitorEvent,
        data: Dict[str, Any],
        severity: str = "info",
    ) -> None:
        """Record a monitoring event."""
        event = MonitorEventRecord(
            event_type=event_type,
            rank=self.rank,
            data=data,
            severity=severity,
        )

        with self._events_lock:
            self._events.append(event)
            # Trim if needed
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        # Invoke callback
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

        # Log based on severity
        if severity == "warning":
            logger.warning(f"Monitor event: {event_type.value} - {data}")
        elif severity == "error":
            logger.error(f"Monitor event: {event_type.value} - {data}")
        else:
            logger.debug(f"Monitor event: {event_type.value} - {data}")

    def _write_final_log(self) -> None:
        """Write final monitoring log to file."""
        if not self.log_dir:
            return

        try:
            log_file = os.path.join(
                self.log_dir,
                f"rank_{self.rank}_monitor_log.json",
            )

            log_data = {
                "rank": self.rank,
                "world_size": self.world_size,
                "hostname": socket.gethostname(),
                "stats": self._stats,
                "final_state": self._state.to_dict(),
                "events": [e.to_dict() for e in self._events[-100:]],  # Last 100 events
            }

            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2, default=str)

            logger.info(f"Monitor log written to {log_file}")

        except Exception as e:
            logger.error(f"Failed to write monitor log: {e}")

    def __enter__(self) -> "InProcessMonitor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_val:
            self.record_fault(str(exc_val), exc_val)
        self.stop()


class DistributedMonitor:
    """Coordinator for monitoring across all ranks.

    This class is intended to run on a coordinator process (e.g., rank 0)
    to aggregate monitoring data from all ranks.
    """

    def __init__(
        self,
        world_size: int,
        heartbeat_config: HeartbeatConfig = None,
        on_timeout: Callable[[int], None] = None,
        on_unhealthy: Callable[[int, str], None] = None,
    ):
        """Initialize distributed monitor.

        Args:
            world_size: Total number of ranks
            heartbeat_config: Heartbeat configuration
            on_timeout: Callback when a rank times out
            on_unhealthy: Callback when a rank becomes unhealthy
        """
        self.world_size = world_size
        self.heartbeat_config = heartbeat_config or HeartbeatConfig()

        self._heartbeat_monitor = HeartbeatMonitor(
            world_size=world_size,
            config=heartbeat_config,
            on_timeout=on_timeout,
        )

        self._rank_states: Dict[int, RankState] = {}
        self._lock = threading.Lock()

        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self.on_timeout = on_timeout
        self.on_unhealthy = on_unhealthy

    def start(self) -> None:
        """Start the distributed monitor."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="DistributedMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

        logger.info(f"DistributedMonitor started for {self.world_size} ranks")

    def stop(self) -> None:
        """Stop the distributed monitor."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        logger.info("DistributedMonitor stopped")

    def record_heartbeat(self, rank: int, record: HeartbeatRecord) -> None:
        """Record a heartbeat from a rank.

        Args:
            rank: The rank that sent the heartbeat
            record: The heartbeat record
        """
        self._heartbeat_monitor.record_heartbeat(rank, record)

    def update_rank_state(self, rank: int, state: RankState) -> None:
        """Update state for a rank.

        Args:
            rank: The rank to update
            state: The new state
        """
        with self._lock:
            self._rank_states[rank] = state

    def get_status(self) -> Dict[str, Any]:
        """Get overall monitoring status."""
        heartbeat_status = self._heartbeat_monitor.get_status()

        with self._lock:
            rank_health = {}
            for rank, state in self._rank_states.items():
                rank_health[rank] = {
                    "mode": state.mode.value,
                    "health_status": state.health_status.value,
                    "fault_count": state.fault_count,
                }

        return {
            "world_size": self.world_size,
            "heartbeat_status": heartbeat_status,
            "rank_health": rank_health,
            "healthy_ranks": self._heartbeat_monitor.get_healthy_ranks(),
            "timed_out_ranks": self._heartbeat_monitor.get_timed_out_ranks(),
        }

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        check_interval = self.heartbeat_config.interval

        while self._running and not self._stop_event.is_set():
            try:
                # Check for timeouts
                timed_out = self._heartbeat_monitor.check_timeouts()

                if timed_out:
                    logger.warning(f"Timed out ranks: {timed_out}")

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            self._stop_event.wait(timeout=check_interval)

    def __enter__(self) -> "DistributedMonitor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
