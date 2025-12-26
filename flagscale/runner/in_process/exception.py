"""Custom exceptions for in-process monitoring.

This module defines exception classes for the in-process monitoring system.
Currently focused on monitoring only (no fault handling/restart logic).
"""

import logging

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
