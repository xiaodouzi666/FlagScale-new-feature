"""State management for in-process monitoring.

This module provides state tracking for distributed training ranks,
inspired by NVIDIA's nvidia-resiliency-ext state management.
"""

import os
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RankMode(Enum):
    """Operational mode of a rank."""

    INITIALIZED = "initialized"  # Pre-assignment state
    ACTIVE = "active"  # Executing wrapped function
    INACTIVE = "inactive"  # Idle/waiting
    UNHEALTHY = "unhealthy"  # Health check failed
    TERMINATED = "terminated"  # Shut down


class HealthStatus(Enum):
    """Health status of a rank."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CHECKING = "checking"


@dataclass
class RankState:
    """State of a single rank in distributed training.

    Attributes:
        rank: Current rank index
        initial_rank: Original rank index (before any reassignment)
        world_size: Total number of ranks
        local_rank: Local rank on this node
        node_rank: Node index in the cluster
        mode: Current operational mode
        health_status: Current health status
        iteration: Current training iteration
        last_heartbeat: Timestamp of last heartbeat
        fault_count: Number of faults detected on this rank
        metrics: Additional metrics/metadata
    """

    rank: int = 0
    initial_rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    node_rank: int = 0
    mode: RankMode = RankMode.INITIALIZED
    health_status: HealthStatus = HealthStatus.UNKNOWN
    iteration: int = 0
    last_heartbeat: float = 0.0
    fault_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None

    @classmethod
    def from_env(cls) -> "RankState":
        """Initialize state from environment variables.

        Reads standard distributed training environment variables:
        - RANK, WORLD_SIZE
        - LOCAL_RANK, LOCAL_WORLD_SIZE
        - NODE_RANK, NNODES
        """
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0)))

        return cls(
            rank=rank,
            initial_rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            node_rank=node_rank,
            mode=RankMode.INITIALIZED,
            health_status=HealthStatus.UNKNOWN,
            last_heartbeat=time.time(),
        )

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def increment_fault_count(self) -> int:
        """Increment and return the fault count."""
        self.fault_count += 1
        return self.fault_count

    def reset_fault_count(self) -> None:
        """Reset the fault count to zero."""
        self.fault_count = 0

    def set_mode(self, mode: RankMode) -> None:
        """Set the operational mode."""
        self.mode = mode

    def set_health_status(
        self, status: HealthStatus, reason: Optional[str] = None
    ) -> None:
        """Set the health status with optional reason."""
        self.health_status = status
        if reason:
            self.metrics["health_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "rank": self.rank,
            "initial_rank": self.initial_rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "mode": self.mode.value,
            "health_status": self.health_status.value,
            "iteration": self.iteration,
            "last_heartbeat": self.last_heartbeat,
            "fault_count": self.fault_count,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankState":
        """Create state from dictionary."""
        state = cls(
            rank=data.get("rank", 0),
            initial_rank=data.get("initial_rank", 0),
            world_size=data.get("world_size", 1),
            local_rank=data.get("local_rank", 0),
            node_rank=data.get("node_rank", 0),
            iteration=data.get("iteration", 0),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            fault_count=data.get("fault_count", 0),
            metrics=data.get("metrics", {}),
        )

        # Parse enums
        if "mode" in data:
            state.mode = RankMode(data["mode"])
        if "health_status" in data:
            state.health_status = HealthStatus(data["health_status"])

        return state


@dataclass(frozen=True)
class FrozenRankState:
    """Immutable snapshot of rank state.

    Used for passing state to callbacks without allowing modification.
    """

    rank: int
    initial_rank: int
    world_size: int
    local_rank: int
    node_rank: int
    mode: RankMode
    health_status: HealthStatus
    iteration: int
    last_heartbeat: float
    fault_count: int
    metrics: tuple  # Frozen version of metrics dict

    @classmethod
    def from_state(cls, state: RankState) -> "FrozenRankState":
        """Create frozen state from mutable state."""
        # Convert metrics dict to tuple of tuples for immutability
        frozen_metrics = tuple(sorted(state.metrics.items()))
        return cls(
            rank=state.rank,
            initial_rank=state.initial_rank,
            world_size=state.world_size,
            local_rank=state.local_rank,
            node_rank=state.node_rank,
            mode=state.mode,
            health_status=state.health_status,
            iteration=state.iteration,
            last_heartbeat=state.last_heartbeat,
            fault_count=state.fault_count,
            metrics=frozen_metrics,
        )

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get a metric value by key."""
        for k, v in self.metrics:
            if k == key:
                return v
        return default


class StateManager:
    """Manager for tracking states of all ranks.

    This class maintains the state of all ranks in a distributed training job,
    providing methods to update and query rank states.
    """

    def __init__(self, world_size: int = 1):
        """Initialize the state manager.

        Args:
            world_size: Total number of ranks to track
        """
        self.world_size = world_size
        self._states: Dict[int, RankState] = {}
        self._local_state: Optional[RankState] = None

    def initialize_local(self) -> RankState:
        """Initialize local rank state from environment."""
        self._local_state = RankState.from_env()
        self._states[self._local_state.rank] = self._local_state
        return self._local_state

    def get_local_state(self) -> Optional[RankState]:
        """Get the local rank's state."""
        return self._local_state

    def get_state(self, rank: int) -> Optional[RankState]:
        """Get state for a specific rank."""
        return self._states.get(rank)

    def set_state(self, rank: int, state: RankState) -> None:
        """Set state for a specific rank."""
        self._states[rank] = state

    def update_state(self, rank: int, **kwargs) -> None:
        """Update specific fields of a rank's state."""
        if rank not in self._states:
            self._states[rank] = RankState(rank=rank)

        state = self._states[rank]
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

    def get_all_states(self) -> Dict[int, RankState]:
        """Get states of all known ranks."""
        return self._states.copy()

    def get_healthy_ranks(self) -> list:
        """Get list of ranks with healthy status."""
        return [
            rank
            for rank, state in self._states.items()
            if state.health_status == HealthStatus.HEALTHY
        ]

    def get_unhealthy_ranks(self) -> list:
        """Get list of ranks with unhealthy status."""
        return [
            rank
            for rank, state in self._states.items()
            if state.health_status == HealthStatus.UNHEALTHY
        ]

    def get_active_ranks(self) -> list:
        """Get list of active ranks."""
        return [
            rank
            for rank, state in self._states.items()
            if state.mode == RankMode.ACTIVE
        ]

    def summary(self) -> Dict[str, Any]:
        """Get summary of all rank states."""
        total = len(self._states)
        healthy = len(self.get_healthy_ranks())
        unhealthy = len(self.get_unhealthy_ranks())
        active = len(self.get_active_ranks())

        return {
            "total_ranks": total,
            "healthy_ranks": healthy,
            "unhealthy_ranks": unhealthy,
            "active_ranks": active,
            "unknown_ranks": total - healthy - unhealthy,
        }
