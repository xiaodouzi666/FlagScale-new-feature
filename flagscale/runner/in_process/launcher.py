"""Launcher for in-process monitoring integration with FlagScale.

This module provides integration points for starting in-process monitoring
through FlagScale's standard configuration and launch system.

Usage via FlagScale:
    python run.py \\
        --config-path ./examples/llama2/conf \\
        --config-name train \\
        action=run \\
        +experiment.runner.enable_in_process_monitoring=true \\
        +experiment.runner.in_process.heartbeat_interval=10.0 \\
        +experiment.runner.in_process.health_check_interval=60.0

Usage in training script:
    from flagscale.runner.in_process.launcher import (
        setup_in_process_monitoring,
        get_wrapper,
    )

    # At start of training
    setup_in_process_monitoring()

    # In training loop
    wrapper = get_wrapper()
    if wrapper:
        wrapper.ping(iteration=step)

    # Checkpoint handling
    if wrapper:
        wrapper.enter_checkpoint_phase()
    save_checkpoint()
    if wrapper:
        wrapper.exit_checkpoint_phase()
"""

import logging
import os

from typing import Any, Dict, Optional, TYPE_CHECKING

# Optional dependency - only needed when integrating with FlagScale config
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    DictConfig = Any  # type: ignore

from .wrap import Wrapper, WrapperConfig

logger = logging.getLogger(__name__)

# Global wrapper instance
_wrapper: Optional[Wrapper] = None


def setup_in_process_monitoring(
    config: DictConfig = None,
    heartbeat_interval: float = 10.0,
    heartbeat_timeout: float = 60.0,
    health_check_interval: float = 60.0,
    enable_cuda_health_check: bool = True,
    enable_nvml_health_check: bool = True,
    enable_network_health_check: bool = False,
    max_rank_faults: int = 5,
    log_dir: str = None,
    # Restart-related parameters
    enable_restart: bool = False,
    max_restarts: int = 3,
    min_world_size: int = 1,
    restart_on_exception: bool = True,
) -> Wrapper:
    """Setup in-process monitoring from configuration or parameters.

    This function should be called at the start of training to initialize
    the monitoring system.

    Args:
        config: FlagScale configuration (DictConfig). If provided, reads
                settings from experiment.runner.in_process
        heartbeat_interval: Interval between heartbeats (seconds)
        heartbeat_timeout: Timeout for heartbeat detection (seconds)
        health_check_interval: Interval between health checks (seconds)
        enable_cuda_health_check: Enable CUDA/GPU health checking
        enable_nvml_health_check: Enable NVML-based GPU monitoring
        enable_network_health_check: Enable network interface checking
        max_rank_faults: Maximum faults before warning
        log_dir: Directory for monitoring logs
        enable_restart: Enable automatic restart on fault detection
        max_restarts: Maximum number of restart attempts
        min_world_size: Minimum world size to continue restarting
        restart_on_exception: Restart on uncaught exceptions

    Returns:
        Started Wrapper instance
    """
    global _wrapper

    if _wrapper is not None:
        logger.warning("In-process monitoring already initialized")
        return _wrapper

    # Build config from FlagScale config if provided
    wrapper_config = WrapperConfig(
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout,
        health_check_interval=health_check_interval,
        enable_cuda_health_check=enable_cuda_health_check,
        enable_nvml_health_check=enable_nvml_health_check,
        enable_network_health_check=enable_network_health_check,
        max_rank_faults=max_rank_faults,
        log_dir=log_dir,
        # Restart-related config
        enable_restart=enable_restart,
        max_restarts=max_restarts,
        min_world_size=min_world_size,
        restart_on_exception=restart_on_exception,
    )

    if config is not None:
        # Extract in_process config from FlagScale config
        in_process_config = _extract_in_process_config(config)
        if in_process_config:
            wrapper_config = _merge_configs(wrapper_config, in_process_config)

    # Create and start wrapper
    _wrapper = Wrapper(config=wrapper_config)
    _wrapper.start()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(
        f"In-process monitoring started for rank {rank}/{world_size} "
        f"(heartbeat: {wrapper_config.heartbeat_interval}s, "
        f"health_check: {wrapper_config.health_check_interval}s)"
    )

    return _wrapper


def _extract_in_process_config(config) -> Optional[Dict[str, Any]]:
    """Extract in-process monitoring config from FlagScale config.

    Args:
        config: FlagScale DictConfig

    Returns:
        Dictionary with in-process config, or None
    """
    try:
        runner_config = config.experiment.runner
        if not runner_config.get("enable_in_process_monitoring", False):
            return None

        in_process = runner_config.get("in_process", {})
        if HAS_OMEGACONF and hasattr(in_process, '_iter_ex'):
            # It's a DictConfig
            return OmegaConf.to_container(in_process, resolve=True)
        return dict(in_process) if in_process else {}

    except Exception as e:
        logger.warning(f"Failed to extract in-process config: {e}")
        return None


def _merge_configs(
    base: WrapperConfig,
    override: Dict[str, Any],
) -> WrapperConfig:
    """Merge override dictionary into base config.

    Args:
        base: Base WrapperConfig
        override: Dictionary with override values

    Returns:
        Merged WrapperConfig
    """
    for key, value in override.items():
        if hasattr(base, key) and value is not None:
            setattr(base, key, value)
    return base


def shutdown_in_process_monitoring() -> None:
    """Shutdown in-process monitoring.

    Call this at the end of training to cleanup monitoring resources.
    """
    global _wrapper

    if _wrapper is not None:
        _wrapper.stop()
        _wrapper = None
        logger.info("In-process monitoring stopped")


def get_wrapper() -> Optional[Wrapper]:
    """Get the global Wrapper instance.

    Returns:
        Wrapper instance or None if not initialized
    """
    return _wrapper


def is_monitoring_enabled() -> bool:
    """Check if in-process monitoring is enabled.

    Returns:
        True if monitoring is running
    """
    return _wrapper is not None and _wrapper._started


def ping(
    iteration: int = None,
    phase: str = None,
    metrics: Dict[str, Any] = None,
) -> None:
    """Send heartbeat ping (convenience function).

    Args:
        iteration: Current training iteration
        phase: Current training phase ("training", "checkpoint", etc.)
        metrics: Additional metrics to record
    """
    if _wrapper is None:
        return

    from .heartbeat import HeartbeatPhase

    hb_phase = None
    if phase:
        phase_map = {
            "training": HeartbeatPhase.TRAINING,
            "checkpoint": HeartbeatPhase.CHECKPOINT,
            "initialization": HeartbeatPhase.INITIALIZATION,
            "evaluation": HeartbeatPhase.EVALUATION,
        }
        hb_phase = phase_map.get(phase.lower())

    _wrapper.ping(iteration=iteration, phase=hb_phase, metrics=metrics)


def enter_checkpoint_phase() -> None:
    """Signal entering checkpoint phase (convenience function)."""
    if _wrapper:
        _wrapper.enter_checkpoint_phase()


def exit_checkpoint_phase() -> None:
    """Signal exiting checkpoint phase (convenience function)."""
    if _wrapper:
        _wrapper.exit_checkpoint_phase()


def record_fault(reason: str, error: Exception = None) -> int:
    """Record a fault (convenience function).

    Args:
        reason: Description of the fault
        error: Associated exception

    Returns:
        Current fault count
    """
    if _wrapper:
        return _wrapper.record_fault(reason, error)
    return 0


# Integration hook for training scripts
def maybe_init_in_process_monitoring(config: DictConfig = None) -> Optional[Wrapper]:
    """Initialize in-process monitoring if enabled in config.

    This function checks the config and only initializes monitoring if
    enable_in_process_monitoring is True.

    Args:
        config: FlagScale configuration

    Returns:
        Wrapper instance if enabled, None otherwise
    """
    if config is None:
        return None

    try:
        enable = config.experiment.runner.get("enable_in_process_monitoring", False)
        if not enable:
            return None

        return setup_in_process_monitoring(config=config)

    except Exception as e:
        logger.warning(f"Failed to initialize in-process monitoring: {e}")
        return None


# Environment variable based initialization
def init_from_env() -> Optional[Wrapper]:
    """Initialize in-process monitoring from environment variables.

    Environment variables:
        FLAGSCALE_IN_PROCESS_MONITORING: Set to "1" or "true" to enable
        FLAGSCALE_HEARTBEAT_INTERVAL: Heartbeat interval in seconds
        FLAGSCALE_HEALTH_CHECK_INTERVAL: Health check interval in seconds
        FLAGSCALE_ENABLE_CUDA_CHECK: Enable CUDA health check ("1" or "true")
        FLAGSCALE_ENABLE_NVML_CHECK: Enable NVML health check ("1" or "true")
        FLAGSCALE_MONITOR_LOG_DIR: Directory for monitoring logs

        # Restart-related environment variables
        FLAGSCALE_ENABLE_RESTART: Enable automatic restart on fault ("1" or "true")
        FLAGSCALE_MAX_RESTARTS: Maximum number of restart attempts
        FLAGSCALE_MIN_WORLD_SIZE: Minimum world size to continue restarting
        FLAGSCALE_RESTART_ON_EXCEPTION: Restart on uncaught exceptions ("1" or "true")

    Returns:
        Wrapper instance if enabled, None otherwise
    """
    enable = os.environ.get("FLAGSCALE_IN_PROCESS_MONITORING", "").lower()
    if enable not in ("1", "true", "yes"):
        return None

    def get_bool(key: str, default: bool = True) -> bool:
        val = os.environ.get(key, "").lower()
        if val in ("0", "false", "no"):
            return False
        if val in ("1", "true", "yes"):
            return True
        return default

    def get_float(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, default))
        except ValueError:
            return default

    def get_int(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, default))
        except ValueError:
            return default

    return setup_in_process_monitoring(
        heartbeat_interval=get_float("FLAGSCALE_HEARTBEAT_INTERVAL", 10.0),
        health_check_interval=get_float("FLAGSCALE_HEALTH_CHECK_INTERVAL", 60.0),
        enable_cuda_health_check=get_bool("FLAGSCALE_ENABLE_CUDA_CHECK", True),
        enable_nvml_health_check=get_bool("FLAGSCALE_ENABLE_NVML_CHECK", True),
        log_dir=os.environ.get("FLAGSCALE_MONITOR_LOG_DIR"),
        # Restart-related options
        enable_restart=get_bool("FLAGSCALE_ENABLE_RESTART", False),
        max_restarts=get_int("FLAGSCALE_MAX_RESTARTS", 3),
        min_world_size=get_int("FLAGSCALE_MIN_WORLD_SIZE", 1),
        restart_on_exception=get_bool("FLAGSCALE_RESTART_ON_EXCEPTION", True),
    )
