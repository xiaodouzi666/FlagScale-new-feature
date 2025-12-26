"""In-process monitoring module for distributed training fault detection.

This module provides monitoring capabilities for distributed training,
inspired by NVIDIA's nvidia-resiliency-ext and AWS's sagemaker-hyperpod
checkpointless training. Currently focused on monitoring only (alerts
and metrics collection, no fault handling/restart logic).

Key Features:
- Heartbeat-based progress monitoring
- Device/network health checks (CUDA, GPU, network interfaces)
- Fault counting and tracking
- Distributed state management
- Event logging and metrics collection

Example Usage:
    # Simple usage with context manager
    from flagscale.runner.in_process import InProcessMonitor

    with InProcessMonitor() as monitor:
        for step, batch in enumerate(dataloader):
            train_step(batch)
            monitor.ping(iteration=step)

    # Using RankMonitorClient API (NVIDIA-style)
    from flagscale.runner.in_process import RankMonitorClient

    client = RankMonitorClient.init_workload_monitoring()
    for step, batch in enumerate(dataloader):
        train_step(batch)
        client.send_heartbeat(iteration=step)
    client.stop()

    # Running health checks
    from flagscale.runner.in_process import (
        HealthCheckRunner,
        CudaHealthCheck,
        NvmlHealthCheck,
    )

    runner = HealthCheckRunner([
        CudaHealthCheck(),
        NvmlHealthCheck(),
    ])
    results = runner.run_all(state)
    for result in results:
        print(f"{result.check_name}: {'OK' if result.healthy else 'FAIL'}")
"""

# State management
from .state import (
    FrozenRankState,
    HealthStatus,
    RankMode,
    RankState,
    StateManager,
)

# Health checks
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

# Heartbeat monitoring
from .heartbeat import (
    HeartbeatConfig,
    HeartbeatMonitor,
    HeartbeatPhase,
    HeartbeatRecord,
    HeartbeatSender,
    RankMonitorClient,
    calculate_and_set_hb_timeouts,
)

# Main monitor
from .monitor import (
    DistributedMonitor,
    InProcessMonitor,
    MonitorEvent,
    MonitorEventRecord,
)

# Composition utilities
from .compose import (
    Compose,
    Pipeline,
    TypedCompose,
    compose,
    find_common_ancestor,
    pipeline,
)

# Exceptions
from .exception import (
    ConfigurationError,
    FaultCounterExceeded,
    HeartbeatTimeoutError,
    HealthCheckError,
    MonitorError,
    StoreError,
)

# Wrapper
from .wrap import (
    CallWrapper,
    Wrapper,
    WrapperConfig,
    init_in_process_monitoring,
    shutdown_in_process_monitoring,
    wrap_training,
)

# Launcher (FlagScale integration)
from .launcher import (
    enter_checkpoint_phase,
    exit_checkpoint_phase,
    get_wrapper,
    init_from_env,
    is_monitoring_enabled,
    maybe_init_in_process_monitoring,
    ping,
    record_fault,
    setup_in_process_monitoring,
    shutdown_in_process_monitoring,
)

__all__ = [
    # State
    "RankState",
    "FrozenRankState",
    "RankMode",
    "HealthStatus",
    "StateManager",
    # Health checks
    "HealthCheck",
    "HealthCheckResult",
    "HealthCheckRunner",
    "CudaHealthCheck",
    "NvmlHealthCheck",
    "NetworkHealthCheck",
    "ChainedHealthCheck",
    "FaultCounter",
    # Heartbeat
    "HeartbeatConfig",
    "HeartbeatPhase",
    "HeartbeatRecord",
    "HeartbeatSender",
    "HeartbeatMonitor",
    "RankMonitorClient",
    "calculate_and_set_hb_timeouts",
    # Monitor
    "InProcessMonitor",
    "DistributedMonitor",
    "MonitorEvent",
    "MonitorEventRecord",
    # Compose
    "Compose",
    "TypedCompose",
    "Pipeline",
    "compose",
    "pipeline",
    "find_common_ancestor",
    # Exceptions
    "MonitorError",
    "HealthCheckError",
    "HeartbeatTimeoutError",
    "StoreError",
    "ConfigurationError",
    "FaultCounterExceeded",
    # Wrapper
    "Wrapper",
    "WrapperConfig",
    "CallWrapper",
    "wrap_training",
    "init_in_process_monitoring",
    "shutdown_in_process_monitoring",
    # Launcher (FlagScale integration)
    "setup_in_process_monitoring",
    "maybe_init_in_process_monitoring",
    "get_wrapper",
    "is_monitoring_enabled",
    "ping",
    "enter_checkpoint_phase",
    "exit_checkpoint_phase",
    "record_fault",
    "init_from_env",
]

__version__ = "0.1.0"
