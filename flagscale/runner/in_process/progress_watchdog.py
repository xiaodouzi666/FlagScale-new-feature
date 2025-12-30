"""Progress watchdog for detecting hung processes.

This module provides a ProgressWatchdog that uses Python's Py_AddPendingCall
mechanism to detect when the main thread is hung (e.g., stuck in NCCL
communication). Inspired by NVIDIA's nvidia-resiliency-ext implementation.

The key insight is that Py_AddPendingCall registers a callback that will be
executed at the next "safe point" in the Python interpreter. If the main
thread is stuck in C code (like NCCL), the callback won't execute, and we
can detect the hang by checking if the timestamp was updated.
"""

import ctypes
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Maximum number of pending calls to track
MAX_PENDING = 1024


@dataclass
class Timestamp:
    """Tracks automatic (via pending call) and manual timestamps.

    Attributes:
        auto: Timestamp updated by Py_AddPendingCall callback
        manual: Timestamp updated by explicit ping() calls
    """
    auto: float = 0.0
    manual: float = 0.0

    def is_expired(self, timeout: float) -> bool:
        """Check if both timestamps have exceeded the timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if both auto and manual timestamps are expired
        """
        now = time.time()
        auto_expired = (now - self.auto) > timeout if self.auto > 0 else True
        manual_expired = (now - self.manual) > timeout if self.manual > 0 else True
        return auto_expired and manual_expired

    def get_age(self) -> float:
        """Get the age of the most recent timestamp."""
        latest = max(self.auto, self.manual)
        if latest > 0:
            return time.time() - latest
        return float('inf')


# Type for the pending call callback
PENDING_CALL_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class ProgressWatchdog:
    """Watchdog that detects hung processes using Py_AddPendingCall.

    This watchdog runs in a separate thread and periodically schedules
    pending calls to the main thread. If the main thread is responsive,
    the pending calls will execute and update a timestamp. If the main
    thread is hung, the timestamp won't update, and we can detect the hang.

    Example:
        watchdog = ProgressWatchdog(timeout=60.0)
        watchdog.start()

        # In training loop
        for step in range(steps):
            train_step()
            watchdog.ping()  # Optional manual ping

        watchdog.shutdown()
    """

    def __init__(
        self,
        timeout: float = 60.0,
        check_interval: float = 1.0,
        on_hang_detected: Optional[Callable[[float], None]] = None,
        on_progress: Optional[Callable[[], None]] = None,
    ):
        """Initialize the progress watchdog.

        Args:
            timeout: Time in seconds after which to consider the process hung
            check_interval: Interval between scheduling pending calls
            on_hang_detected: Callback when hang is detected (receives age in seconds)
            on_progress: Callback when progress is detected
        """
        self.timeout = timeout
        self.check_interval = check_interval
        self.on_hang_detected = on_hang_detected
        self.on_progress = on_progress

        self._timestamp = Timestamp()
        self._lock = threading.Lock()

        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()

        # Track pending calls
        self._scheduled_count = 0
        self._completed_count = 0

        # Create the callback function
        self._callback = PENDING_CALL_FUNC(self._pending_call_callback)

        # Track if we've already reported a hang
        self._hang_reported = False

    def _pending_call_callback(self, arg: ctypes.c_void_p) -> int:
        """Callback executed by Python interpreter in the main thread.

        This function is called via Py_AddPendingCall when the Python
        interpreter reaches a safe point. If this executes, it means
        the main thread is not hung.

        Args:
            arg: Unused argument (required by Py_AddPendingCall signature)

        Returns:
            0 to indicate success (required by Py_AddPendingCall)
        """
        with self._lock:
            self._timestamp.auto = time.time()
            self._completed_count = (self._completed_count + 1) % MAX_PENDING

        if self._hang_reported:
            self._hang_reported = False
            logger.info("Process recovered from hang state")

        if self.on_progress:
            try:
                self.on_progress()
            except Exception as e:
                logger.warning(f"Error in on_progress callback: {e}")

        return 0

    def _schedule_pending_call(self) -> bool:
        """Schedule a pending call to the main thread.

        Returns:
            True if successfully scheduled, False otherwise
        """
        try:
            # Check if Python is finalizing
            if hasattr(ctypes.pythonapi, 'Py_IsInitialized'):
                if not ctypes.pythonapi.Py_IsInitialized():
                    return False

            result = ctypes.pythonapi.Py_AddPendingCall(self._callback, None)
            if result == 0:
                with self._lock:
                    self._scheduled_count = (self._scheduled_count + 1) % MAX_PENDING
                return True
            else:
                logger.warning("Py_AddPendingCall returned non-zero")
                return False

        except Exception as e:
            logger.warning(f"Failed to schedule pending call: {e}")
            return False

    def _watchdog_loop(self) -> None:
        """Main watchdog loop running in separate thread."""
        logger.debug("Progress watchdog loop started")

        while self._running and not self._stop_event.is_set():
            # Check if paused
            if self._paused:
                self._pause_event.set()
                self._resume_event.wait()
                self._resume_event.clear()
                continue

            # Schedule a pending call
            self._schedule_pending_call()

            # Check for hang
            with self._lock:
                if self._timestamp.is_expired(self.timeout):
                    age = self._timestamp.get_age()
                    if not self._hang_reported:
                        self._hang_reported = True
                        logger.warning(
                            f"Hang detected: no progress for {age:.1f}s "
                            f"(timeout: {self.timeout:.1f}s)"
                        )
                        if self.on_hang_detected:
                            try:
                                self.on_hang_detected(age)
                            except Exception as e:
                                logger.error(f"Error in on_hang_detected callback: {e}")

            # Wait for next check
            self._stop_event.wait(timeout=self.check_interval)

        logger.debug("Progress watchdog loop ended")

    def start(self) -> None:
        """Start the watchdog thread."""
        if self._running:
            logger.warning("Progress watchdog already running")
            return

        self._running = True
        self._stop_event.clear()
        self._hang_reported = False

        # Initialize timestamp
        with self._lock:
            now = time.time()
            self._timestamp.auto = now
            self._timestamp.manual = now

        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="ProgressWatchdog",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            f"Progress watchdog started (timeout: {self.timeout}s, "
            f"check_interval: {self.check_interval}s)"
        )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the watchdog thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Resume if paused
        if self._paused:
            self._resume_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Progress watchdog thread did not stop in time")

        logger.info("Progress watchdog stopped")

    def ping(self) -> None:
        """Manually update the progress timestamp.

        Call this periodically in the training loop to indicate progress.
        This is complementary to the automatic Py_AddPendingCall mechanism.
        """
        with self._lock:
            self._timestamp.manual = time.time()

        if self._hang_reported:
            self._hang_reported = False
            logger.info("Process recovered from hang state (manual ping)")

    def pause_and_synchronize(self) -> None:
        """Pause the watchdog and wait for pending calls to complete.

        Use this before operations that might trigger false hang detection,
        such as checkpoint saving.
        """
        if not self._running or self._paused:
            return

        self._paused = True
        self._pause_event.clear()
        self._pause_event.wait(timeout=5.0)

        # Wait for pending calls to drain
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._scheduled_count == self._completed_count:
                    break
            time.sleep(0.1)

        logger.debug("Progress watchdog paused")

    def resume(self) -> None:
        """Resume the watchdog after pausing."""
        if not self._paused:
            return

        # Reset timestamp to avoid false hang detection
        with self._lock:
            now = time.time()
            self._timestamp.auto = now
            self._timestamp.manual = now

        self._paused = False
        self._resume_event.set()

        logger.debug("Progress watchdog resumed")

    def get_status(self) -> dict:
        """Get current watchdog status.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "running": self._running,
                "paused": self._paused,
                "timeout": self.timeout,
                "last_auto_timestamp": self._timestamp.auto,
                "last_manual_timestamp": self._timestamp.manual,
                "timestamp_age": self._timestamp.get_age(),
                "is_expired": self._timestamp.is_expired(self.timeout),
                "hang_reported": self._hang_reported,
                "scheduled_calls": self._scheduled_count,
                "completed_calls": self._completed_count,
            }

    def __enter__(self) -> "ProgressWatchdog":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
