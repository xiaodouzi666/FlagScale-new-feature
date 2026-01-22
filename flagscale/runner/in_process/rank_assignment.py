"""Rank assignment/reassignment logic for in-process restart.

This module is responsible for mapping physical ranks to logical ranks,
determining which ranks should be active in a training job, and handling
rank replacement when faults occur.

Inspired by NVIDIA's nvidia-resiliency-ext rank_assignment module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Optional, Set

from .state import RankMode, RankState


@dataclass
class RankAssignmentCtx:
    """Context for rank assignment operations.
    
    Attributes:
        state: The current state of the rank executing the assignment
        terminated_ranks: Set of initial_ranks that have been terminated/faulted
    """
    state: RankState
    terminated_ranks: Set[int]


class RankAssignment(ABC):
    """Abstract base class for rank assignment strategies."""

    @abstractmethod
    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        """Apply rank assignment logic.
        
        Args:
            ctx: Current assignment context
            
        Returns:
            Modified context with updated state
        """
        pass


class SimpleRankAssignment(RankAssignment):
    """Simple rank assignment: Active if not terminated.
    
    This strategy simply marks a rank as active if it's not in the terminated set.
    It does not perform re-indexing or world size adjustment.
    """

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        if ctx.state.initial_rank in ctx.terminated_ranks:
            ctx.state.mode = RankMode.TERMINATED
        else:
            ctx.state.mode = RankMode.ACTIVE
            ctx.state.rank = ctx.state.initial_rank
            ctx.state.active_world_size = ctx.state.world_size
            
        return ctx


class MaxActiveWorldSize(RankAssignment):
    """Limits the number of active ranks and performs replacement.
    
    This strategy selects up to `max_active_world_size` healthy ranks to be ACTIVE.
    Ranks are selected based on their `initial_rank` order.
    
    Logic:
    1. List all available healthy ranks (those not in terminated_ranks).
    2. Sort by initial_rank.
    3. Take the first `active_world_size` ranks.
    4. Assign them logical ranks 0..N-1.
    5. Mark remaining healthy ranks as INACTIVE (spares).
    """

    def __init__(self, max_active_world_size: int):
        self.max_active_world_size = max_active_world_size

    def __call__(self, ctx: RankAssignmentCtx) -> RankAssignmentCtx:
        # Determine the target active world size
        # It's the minimum of configured max, and total available ranks (physically)
        # But we really want to target exactly max_active_world_size if possible
        # for training consistency.
        
        # 1. Identify all potentially available ranks (physically present and not terminated)
        # RankState usually contains world_size which is the physical world size initiated by mpirun/torchrun
        potential_ranks = []
        for r in range(ctx.state.world_size):
            if r not in ctx.terminated_ranks:
                potential_ranks.append(r)
        
        potential_ranks.sort()
        
        # 2. Select the active subset
        active_initial_ranks = potential_ranks[:self.max_active_world_size]
        
        # 3. Determine status of THIS rank
        my_initial_rank = ctx.state.initial_rank
        
        if my_initial_rank in ctx.terminated_ranks:
            ctx.state.mode = RankMode.TERMINATED
            ctx.state.rank = -1 # Invalid
        elif my_initial_rank in active_initial_ranks:
            # I am active!
            # Find my logical rank index
            logical_rank = active_initial_ranks.index(my_initial_rank)
            
            ctx.state.mode = RankMode.ACTIVE
            ctx.state.rank = logical_rank
            ctx.state.active_world_size = len(active_initial_ranks)
        else:
            # I am a healthy spare
            ctx.state.mode = RankMode.INACTIVE
            ctx.state.rank = -1 # Invalid as I'm not in the process group
            ctx.state.active_world_size = len(active_initial_ranks)
            
        return ctx
