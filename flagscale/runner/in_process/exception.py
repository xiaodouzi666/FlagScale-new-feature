"""Custom exceptions for in-process monitoring.

This module defines exception classes for the in-process monitoring system,
including exceptions for fault detection and restart triggering.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class MonitorError(Exception):
    """Base exception for monitoring-related errors."""
    pass


class HealthCheckError(MonitorError):
    """Raised when a health check fails.

    In monitoring-only mode, this exception is caught and logged,
    not used to trigger restarts.
    """

    def __init__(self, message: str, check_name: str = None, details: dict = None):
        super().__init__(message)
        self.check_name = check_name
        self.details = details or {}

    def __str__(self):
        base = super().__str__()
        if self.check_name:
            return f"[{self.check_name}] {base}"
        return base


class HeartbeatTimeoutError(MonitorError):
    """Raised when heartbeat timeout is detected.

    In monitoring-only mode, this is logged as a warning,
    not used to trigger restarts.
    """

    def __init__(self, rank: int, last_heartbeat: float, timeout: float):
        self.rank = rank
        self.last_heartbeat = last_heartbeat
        self.timeout = timeout
        message = (
            f"Heartbeat timeout for rank {rank}: "
            f"last heartbeat was {last_heartbeat:.2f}s ago, timeout is {timeout:.2f}s"
        )
        super().__init__(message)


class StoreError(MonitorError):
    """Raised when distributed store operations fail."""
    pass


class ConfigurationError(MonitorError):
    """Raised when configuration is invalid."""
    pass


class FaultCounterExceeded(MonitorError):
    """Raised when fault counter exceeds the threshold.

    In monitoring-only mode, this is logged but does not trigger process exit.
    """

    def __init__(self, rank: int, fault_count: int, max_faults: int):
        self.rank = rank
        self.fault_count = fault_count
        self.max_faults = max_faults
        message = (
            f"Fault counter exceeded for rank {rank}: "
            f"{fault_count} faults (max: {max_faults})"
        )
        super().__init__(message)


# =============================================================================
# Restart-related exceptions
# =============================================================================


class RankShouldRestart(MonitorError):
    """Raised when a rank should restart due to detected fault.

    This exception is used to signal the restart loop that the current
    iteration has failed and should be retried.

    Attributes:
        rank: The rank that should restart
        reason: Description of why restart is needed
        original_error: The original exception that caused the restart
        fault_type: Type of fault (heartbeat_timeout, health_check, hang, etc.)
    """

    def __init__(
        self,
        reason: str,
        rank: int = None,
        original_error: Exception = None,
        fault_type: str = "unknown",
    ):
        self.rank = rank
        self.reason = reason
        self.original_error = original_error
        self.fault_type = fault_type
        message = f"Rank {rank} should restart: {reason}"
        if original_error:
            message += f" (caused by: {type(original_error).__name__})"
        super().__init__(message)


class RestartAbort(MonitorError):
    """Raised when restart loop should be terminated.

    This exception signals that no more restart attempts should be made,
    typically because max restarts exceeded or world_size too small.

    Attributes:
        reason: Description of why restart is aborted
        restart_count: Number of restarts that were attempted
    """

    def __init__(self, reason: str, restart_count: int = 0):
        self.reason = reason
        self.restart_count = restart_count
        message = f"Restart aborted after {restart_count} attempts: {reason}"
        super().__init__(message)


class HealthCheckPassed(Exception):
    """Raised after successful health check in restart loop.

    This is a control flow exception used to signal that health check
    passed and the restart loop should continue to the next phase.
    Not a subclass of MonitorError as it's not an error condition.
    """

    pass


class RestartRequired(MonitorError):
    """Raised to indicate that a restart is required but not yet triggered.

    This is used internally to collect restart requirements before
    actually triggering a restart.
    """

    def __init__(self, reason: str, severity: str = "warning"):
        self.reason = reason
        self.severity = severity
        super().__init__(reason)
