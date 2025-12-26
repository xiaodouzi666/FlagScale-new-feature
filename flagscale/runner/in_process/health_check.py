"""Health check abstractions for in-process monitoring.

This module provides health check mechanisms inspired by NVIDIA's
nvidia-resiliency-ext HealthCheck abstraction. Currently focused on
monitoring only (outputs healthy/unhealthy + reason + metrics).

Key components:
- HealthCheck: Abstract base class for all health checks
- CudaHealthCheck: Validates GPU/CUDA context health
- ChainedHealthCheck: Combines multiple health checks
- FaultCounter: Tracks fault occurrences (monitoring only)
"""

import datetime
import logging
import os
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exception import FaultCounterExceeded, HealthCheckError
from .state import FrozenRankState, HealthStatus, RankState

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Attributes:
        healthy: Whether the check passed
        check_name: Name of the health check
        reason: Human-readable reason for the result
        metrics: Additional metrics collected during the check
        duration_ms: Time taken to perform the check in milliseconds
        timestamp: When the check was performed
    """

    healthy: bool
    check_name: str
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "healthy": self.healthy,
            "check_name": self.check_name,
            "reason": self.reason,
            "metrics": self.metrics,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class HealthCheck(ABC):
    """Abstract base class for health checks.

    Health checks ensure a worker is in a healthy state. In monitoring-only
    mode, unhealthy states are logged and reported but do not trigger restarts.

    Subclasses should implement the `check` method to perform the actual
    health validation.
    """

    def __init__(self, name: str = None):
        """Initialize the health check.

        Args:
            name: Optional name for this health check. If not provided,
                  uses the class name.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Perform the health check.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult with the check outcome
        """
        raise NotImplementedError

    def __call__(self, state: FrozenRankState) -> HealthCheckResult:
        """Execute the health check with timing.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult with timing information
        """
        start_time = time.time()
        try:
            result = self.check(state)
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check {self.name} failed with exception: {e}")
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason=f"Exception during check: {str(e)}",
                duration_ms=duration_ms,
            )


class CudaHealthCheck(HealthCheck):
    """Validates CUDA/GPU context health.

    Performs GPU synchronization to verify the CUDA context is healthy.
    Uses the device specified by LOCAL_RANK environment variable.
    """

    def __init__(
        self,
        timeout: datetime.timedelta = datetime.timedelta(seconds=30),
        name: str = "CudaHealthCheck",
    ):
        """Initialize CUDA health check.

        Args:
            timeout: Maximum time to wait for GPU synchronization
            name: Name for this health check
        """
        super().__init__(name)
        self.timeout = timeout
        self._torch_available = None
        self._cuda_available = None

    def _check_torch_cuda(self) -> Tuple[bool, bool]:
        """Check if torch and CUDA are available."""
        if self._torch_available is None:
            try:
                import torch

                self._torch_available = True
                self._cuda_available = torch.cuda.is_available()
            except ImportError:
                self._torch_available = False
                self._cuda_available = False
        return self._torch_available, self._cuda_available

    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Check CUDA context health via GPU synchronization.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult indicating GPU health
        """
        torch_available, cuda_available = self._check_torch_cuda()

        if not torch_available:
            return HealthCheckResult(
                healthy=True,  # Not unhealthy, just not applicable
                check_name=self.name,
                reason="PyTorch not available, skipping CUDA check",
                metrics={"torch_available": False},
            )

        if not cuda_available:
            return HealthCheckResult(
                healthy=True,  # Not unhealthy if no CUDA device
                check_name=self.name,
                reason="CUDA not available, skipping GPU check",
                metrics={"cuda_available": False},
            )

        import torch

        # Determine device from LOCAL_RANK
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_count = torch.cuda.device_count()

        if local_rank >= device_count:
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason=f"LOCAL_RANK {local_rank} >= device_count {device_count}",
                metrics={
                    "local_rank": local_rank,
                    "device_count": device_count,
                },
            )

        try:
            device = torch.device(f"cuda:{local_rank}")

            # First sync: wait for any pending kernels
            torch.cuda.synchronize(device)

            # Second sync: validate context health
            torch.cuda.synchronize(device)

            # Collect GPU metrics
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory = torch.cuda.max_memory_allocated(device)

            return HealthCheckResult(
                healthy=True,
                check_name=self.name,
                reason="CUDA context healthy",
                metrics={
                    "device": local_rank,
                    "memory_allocated_mb": memory_allocated / (1024 * 1024),
                    "memory_reserved_mb": memory_reserved / (1024 * 1024),
                    "max_memory_allocated_mb": max_memory / (1024 * 1024),
                },
            )

        except RuntimeError as e:
            error_msg = str(e)
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason=f"CUDA synchronization failed: {error_msg}",
                metrics={
                    "device": local_rank,
                    "error": error_msg,
                },
            )


class NvmlHealthCheck(HealthCheck):
    """Validates GPU health using NVIDIA Management Library (NVML).

    Checks GPU temperature, memory, ECC errors, and other metrics
    via pynvml (if available).
    """

    def __init__(
        self,
        temp_threshold: float = 90.0,  # Celsius
        memory_threshold: float = 0.95,  # 95% usage
        name: str = "NvmlHealthCheck",
    ):
        """Initialize NVML health check.

        Args:
            temp_threshold: Maximum GPU temperature in Celsius
            memory_threshold: Maximum memory usage ratio (0-1)
            name: Name for this health check
        """
        super().__init__(name)
        self.temp_threshold = temp_threshold
        self.memory_threshold = memory_threshold
        self._nvml_available = None

    def _check_nvml(self) -> bool:
        """Check if pynvml is available."""
        if self._nvml_available is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_available = True
            except (ImportError, Exception):
                self._nvml_available = False
        return self._nvml_available

    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Check GPU health via NVML.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult with GPU health metrics
        """
        if not self._check_nvml():
            return HealthCheckResult(
                healthy=True,
                check_name=self.name,
                reason="pynvml not available, skipping NVML check",
                metrics={"nvml_available": False},
            )

        import pynvml

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

            # Get GPU info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_ratio = mem_info.used / mem_info.total

            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except pynvml.NVMLError:
                power = -1

            # ECC errors (if supported)
            try:
                ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                )
            except pynvml.NVMLError:
                ecc_errors = 0

            metrics = {
                "device": local_rank,
                "gpu_name": name,
                "temperature_c": temp,
                "memory_used_gb": mem_info.used / (1024**3),
                "memory_total_gb": mem_info.total / (1024**3),
                "memory_usage_ratio": mem_used_ratio,
                "power_w": power,
                "ecc_errors": ecc_errors,
            }

            # Check thresholds
            issues = []
            if temp > self.temp_threshold:
                issues.append(f"Temperature {temp}C > {self.temp_threshold}C")
            if mem_used_ratio > self.memory_threshold:
                issues.append(
                    f"Memory usage {mem_used_ratio:.1%} > {self.memory_threshold:.1%}"
                )
            if ecc_errors > 0:
                issues.append(f"ECC errors detected: {ecc_errors}")

            if issues:
                return HealthCheckResult(
                    healthy=False,
                    check_name=self.name,
                    reason="; ".join(issues),
                    metrics=metrics,
                )

            return HealthCheckResult(
                healthy=True,
                check_name=self.name,
                reason="GPU healthy",
                metrics=metrics,
            )

        except pynvml.NVMLError as e:
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason=f"NVML error: {str(e)}",
                metrics={"device": local_rank, "error": str(e)},
            )


class NetworkHealthCheck(HealthCheck):
    """Validates network interface health.

    Checks network connectivity and interface status.
    """

    def __init__(
        self,
        interfaces: List[str] = None,
        name: str = "NetworkHealthCheck",
    ):
        """Initialize network health check.

        Args:
            interfaces: List of interface names to check.
                       If None, checks common RDMA interfaces.
            name: Name for this health check
        """
        super().__init__(name)
        self.interfaces = interfaces or ["ib0", "eth0", "enp"]

    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Check network interface health.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult with network health status
        """
        import socket
        import subprocess

        metrics = {}
        issues = []

        # Check if we can resolve hostname
        try:
            hostname = socket.gethostname()
            ip_addr = socket.gethostbyname(hostname)
            metrics["hostname"] = hostname
            metrics["ip_address"] = ip_addr
        except socket.error as e:
            issues.append(f"Cannot resolve hostname: {e}")

        # Check network interfaces (Linux only)
        try:
            result = subprocess.run(
                ["ip", "link", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout
                for iface in self.interfaces:
                    if iface in output:
                        # Check if interface is UP
                        if f"{iface}" in output and "state UP" in output:
                            metrics[f"{iface}_status"] = "UP"
                        elif f"{iface}" in output:
                            metrics[f"{iface}_status"] = "DOWN"
                            issues.append(f"Interface {iface} is DOWN")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ip command not available or timed out
            pass

        if issues:
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason="; ".join(issues),
                metrics=metrics,
            )

        return HealthCheckResult(
            healthy=True,
            check_name=self.name,
            reason="Network interfaces healthy",
            metrics=metrics,
        )


class ChainedHealthCheck(HealthCheck):
    """Chains multiple health checks together.

    Executes all checks and aggregates results. Considers overall health
    as unhealthy if any individual check fails.
    """

    def __init__(
        self,
        checks: List[HealthCheck],
        name: str = "ChainedHealthCheck",
        stop_on_failure: bool = False,
    ):
        """Initialize chained health check.

        Args:
            checks: List of health checks to execute
            name: Name for this health check
            stop_on_failure: If True, stop checking after first failure
        """
        super().__init__(name)
        self.checks = checks
        self.stop_on_failure = stop_on_failure

    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Execute all chained health checks.

        Args:
            state: Current frozen state of the rank

        Returns:
            Aggregated HealthCheckResult
        """
        results = []
        all_healthy = True
        all_metrics = {}
        reasons = []

        for hc in self.checks:
            result = hc(state)
            results.append(result)
            all_metrics[hc.name] = result.metrics

            if not result.healthy:
                all_healthy = False
                reasons.append(f"[{hc.name}] {result.reason}")

                if self.stop_on_failure:
                    break

        return HealthCheckResult(
            healthy=all_healthy,
            check_name=self.name,
            reason="; ".join(reasons) if reasons else "All checks passed",
            metrics={
                "individual_results": [r.to_dict() for r in results],
                "checks_executed": len(results),
                "checks_passed": sum(1 for r in results if r.healthy),
                **all_metrics,
            },
        )


class FaultCounter(HealthCheck):
    """Tracks fault occurrences on the current rank.

    In monitoring-only mode, this logs warnings when faults exceed
    the threshold but does not trigger process termination.
    """

    def __init__(
        self,
        max_rank_faults: int = 3,
        name: str = "FaultCounter",
    ):
        """Initialize fault counter.

        Args:
            max_rank_faults: Maximum faults before raising warning
            name: Name for this health check
        """
        super().__init__(name)
        self.max_rank_faults = max_rank_faults

    def check(self, state: FrozenRankState) -> HealthCheckResult:
        """Check if fault count exceeds threshold.

        Args:
            state: Current frozen state of the rank

        Returns:
            HealthCheckResult indicating fault count status
        """
        fault_count = state.fault_count

        metrics = {
            "fault_count": fault_count,
            "max_faults": self.max_rank_faults,
            "rank": state.rank,
        }

        if fault_count >= self.max_rank_faults:
            logger.warning(
                f"Fault counter exceeded for rank {state.rank}: "
                f"{fault_count} >= {self.max_rank_faults}"
            )
            return HealthCheckResult(
                healthy=False,
                check_name=self.name,
                reason=f"Fault count {fault_count} >= max {self.max_rank_faults}",
                metrics=metrics,
            )

        return HealthCheckResult(
            healthy=True,
            check_name=self.name,
            reason=f"Fault count {fault_count} < max {self.max_rank_faults}",
            metrics=metrics,
        )


class HealthCheckRunner:
    """Runs health checks and collects results.

    This class provides a convenient way to run multiple health checks
    and aggregate their results for monitoring purposes.
    """

    def __init__(self, checks: List[HealthCheck] = None):
        """Initialize the health check runner.

        Args:
            checks: List of health checks to run. If None, uses default checks.
        """
        self.checks = checks or self._default_checks()
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def _default_checks(self) -> List[HealthCheck]:
        """Get default health checks."""
        return [
            CudaHealthCheck(),
            NvmlHealthCheck(),
            NetworkHealthCheck(),
        ]

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks.append(check)

    def run_all(self, state: RankState) -> List[HealthCheckResult]:
        """Run all health checks.

        Args:
            state: Current rank state

        Returns:
            List of health check results
        """
        frozen_state = FrozenRankState.from_state(state)
        results = []

        for check in self.checks:
            result = check(frozen_state)
            results.append(result)

            # Log result
            if result.healthy:
                logger.debug(
                    f"Health check {result.check_name} passed: {result.reason}"
                )
            else:
                logger.warning(
                    f"Health check {result.check_name} failed: {result.reason}"
                )

        # Store in history
        self._add_to_history(results)

        return results

    def _add_to_history(self, results: List[HealthCheckResult]) -> None:
        """Add results to history with size limit."""
        entry = {
            "timestamp": time.time(),
            "results": [r.to_dict() for r in results],
            "all_healthy": all(r.healthy for r in results),
        }
        self._history.append(entry)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get health check history."""
        return self._history.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recent health checks."""
        if not self._history:
            return {"total_checks": 0, "healthy_ratio": 0.0}

        total = len(self._history)
        healthy = sum(1 for entry in self._history if entry["all_healthy"])

        return {
            "total_checks": total,
            "healthy_count": healthy,
            "unhealthy_count": total - healthy,
            "healthy_ratio": healthy / total,
            "last_check": self._history[-1] if self._history else None,
        }
