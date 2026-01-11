"""In-process monitoring module for distributed training fault detection.

This module provides monitoring capabilities for distributed training,
inspired by NVIDIA's nvidia-resiliency-ext and AWS's sagemaker-hyperpod
checkpointless training. Supports both monitoring-only mode and automatic
restart on fault detection.

Key Features:
- Heartbeat-based progress monitoring
- Device/network health checks (CUDA, GPU, network interfaces)
- Fault counting and tracking
- Distributed state management
- Event logging and metrics collection
- Automatic restart on fault detection (optional)
- Configurable retry limits and backoff

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

# Progress watchdog (hang detection)
from .progress_watchdog import (
    ProgressWatchdog,
    Timestamp,
)

# Exceptions
from .exception import (
    ConfigurationError,
    FaultCounterExceeded,
    HeartbeatTimeoutError,
    HealthCheckError,
    HealthCheckPassed,
    MonitorError,
    RankShouldRestart,
    RestartAbort,
    RestartRequired,
    StoreError,
)

# Abort handlers (for restart)
from .abort import (
    Abort,
    AbortCUDA,
    AbortNCCL,
    AbortTorchDistributed,
    ComposedAbort,
    create_default_abort_handler,
)

# Initialize/Retry control (for restart)
from .initialize import (
    ComposedInitialize,
    Initialize,
    InitializeDistributed,
    RestartConfig,
    RetryController,
    create_default_retry_controller,
)

# Restart synchronization (cross-rank coordination)
from .restart_sync import (
    RestartCoordinator,
    create_restart_coordinator,
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
    # Progress watchdog
    "ProgressWatchdog",
    "Timestamp",
    # Exceptions
    "MonitorError",
    "HealthCheckError",
    "HeartbeatTimeoutError",
    "StoreError",
    "ConfigurationError",
    "FaultCounterExceeded",
    "RankShouldRestart",
    "RestartAbort",
    "RestartRequired",
    "HealthCheckPassed",
    # Abort handlers (for restart)
    "Abort",
    "AbortTorchDistributed",
    "AbortNCCL",
    "AbortCUDA",
    "ComposedAbort",
    "create_default_abort_handler",
    # Initialize/Retry control (for restart)
    "Initialize",
    "RetryController",
    "InitializeDistributed",
    "ComposedInitialize",
    "RestartConfig",
    "create_default_retry_controller",
    # Restart synchronization (cross-rank coordination)
    "RestartCoordinator",
    "create_restart_coordinator",
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

__version__ = "0.2.0"  # Added restart support
