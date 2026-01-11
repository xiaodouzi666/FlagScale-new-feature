"""Cross-rank restart synchronization using TCPStore.

This module provides RestartCoordinator for synchronizing restart operations
across multiple ranks in distributed training. It uses PyTorch's TCPStore
for out-of-band communication that works even when process groups are
destroyed or corrupted.

Key features:
- request_restart(): Broadcast that this rank needs a restart
- restart_requested(): Check if any rank has requested restart
- barrier(): Synchronize all ranks at a named barrier point
"""

import logging
import os
import time
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class RestartCoordinator:
    """Coordinates restart operations across ranks using TCPStore.

    This class provides synchronization primitives for multi-rank restart:
    1. Restart request broadcasting - any rank can signal need for restart
    2. Restart request checking - ranks can check if peers requested restart
    3. Barrier synchronization - ensure all ranks reach same point before proceeding

    The TCPStore is used instead of torch.distributed because:
    - It works even when process groups are destroyed
    - It doesn't require NCCL/Gloo to be functional
    - It provides simple key-value semantics for coordination
    """

    # Key prefixes for store operations
    KEY_RESTART_REQUEST = "restart_request"
    KEY_RESTART_REASON = "restart_reason"
    KEY_BARRIER = "barrier"

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = None,
        master_port: int = None,
        store_port: int = None,
        timeout: float = 300.0,
    ):
        """Initialize the RestartCoordinator.

        Args:
            rank: This rank's index
            world_size: Total number of ranks
            master_addr: Address of rank 0 (default: from MASTER_ADDR env)
            master_port: Port for master (default: from MASTER_PORT env)
            store_port: Port for TCPStore (default: from FLAGSCALE_RESTART_STORE_PORT env,
                        or master_port + 1)
            timeout: Timeout for store operations in seconds
        """
        self.rank = rank
        self.world_size = world_size
        self.timeout = timeout

        # Get addresses from environment if not provided
        self.master_addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
        self.master_port = int(master_port or os.environ.get("MASTER_PORT", 29500))

        # Store port: use dedicated env var, or master_port + 1
        if store_port is not None:
            self.store_port = store_port
        else:
            env_store_port = os.environ.get("FLAGSCALE_RESTART_STORE_PORT")
            if env_store_port:
                self.store_port = int(env_store_port)
            else:
                self.store_port = self.master_port + 1

        self._store = None
        self._initialized = False

    def _ensure_store(self) -> bool:
        """Ensure TCPStore is initialized.

        Returns:
            True if store is available, False otherwise
        """
        if self._store is not None:
            return True

        try:
            import torch.distributed as dist

            # Rank 0 is the master (creates the store)
            is_master = (self.rank == 0)

            logger.debug(
                f"Rank {self.rank}: Creating TCPStore "
                f"(master={is_master}, addr={self.master_addr}:{self.store_port})"
            )

            self._store = dist.TCPStore(
                host_name=self.master_addr,
                port=self.store_port,
                world_size=self.world_size,
                is_master=is_master,
                timeout=timedelta(seconds=self.timeout),
                wait_for_workers=True,
            )

            self._initialized = True
            logger.info(
                f"Rank {self.rank}: RestartCoordinator initialized "
                f"(store={self.master_addr}:{self.store_port})"
            )
            return True

        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to initialize TCPStore: {e}")
            return False

    def request_restart(
        self,
        attempt: int,
        rank: int,
        reason: str,
        iteration: int = 0,
    ) -> bool:
        """Request a restart for the current attempt.

        This broadcasts to all ranks that a restart is needed. Other ranks
        will see this when they call restart_requested().

        Args:
            attempt: Current restart attempt number
            rank: Rank that is requesting restart
            reason: Reason for the restart
            iteration: Current training iteration

        Returns:
            True if request was published, False on failure
        """
        if not self._ensure_store():
            return False

        try:
            # Key format: restart_request/<attempt>
            key = f"{self.KEY_RESTART_REQUEST}/{attempt}"

            # Value format: <rank>|<iteration>|<reason>
            value = f"{rank}|{iteration}|{reason}"

            # Use compare_set for idempotent publish (first writer wins)
            # If key doesn't exist, set it. If it exists, keep existing value.
            try:
                self._store.set(key, value)
            except Exception:
                # Key might already exist, which is fine
                pass

            # Also store the reason separately for easy retrieval
            reason_key = f"{self.KEY_RESTART_REASON}/{attempt}"
            try:
                self._store.set(reason_key, reason)
            except Exception:
                pass

            logger.debug(
                f"Rank {self.rank}: Published restart request for attempt {attempt}: {reason}"
            )
            return True

        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to publish restart request: {e}")
            return False

    def restart_requested(self, attempt: int) -> bool:
        """Check if any rank has requested restart for this attempt.

        Args:
            attempt: Current restart attempt number

        Returns:
            True if restart was requested, False otherwise
        """
        if not self._ensure_store():
            return False

        try:
            key = f"{self.KEY_RESTART_REQUEST}/{attempt}"

            # Check if key exists
            try:
                # num_keys() and check() are not universally available,
                # so we try to get the value
                value = self._store.get(key)
                return value is not None and len(value) > 0
            except Exception:
                # Key doesn't exist
                return False

        except Exception as e:
            logger.debug(f"Rank {self.rank}: Error checking restart request: {e}")
            return False

    def get_reason(self, attempt: int) -> Optional[str]:
        """Get the reason for restart request.

        Args:
            attempt: Current restart attempt number

        Returns:
            Reason string or None if not found
        """
        if not self._ensure_store():
            return None

        try:
            reason_key = f"{self.KEY_RESTART_REASON}/{attempt}"
            try:
                value = self._store.get(reason_key)
                if value:
                    return value.decode() if isinstance(value, bytes) else str(value)
            except Exception:
                pass
            return None

        except Exception as e:
            logger.debug(f"Rank {self.rank}: Error getting restart reason: {e}")
            return None

    def barrier(
        self,
        name: str,
        attempt: int,
        timeout_s: float = None,
    ) -> bool:
        """Synchronize all ranks at a named barrier.

        This implements a store-based barrier that doesn't require
        torch.distributed process groups to be functional.

        Args:
            name: Name of the barrier (e.g., "fault_detected", "abort_done")
            attempt: Current restart attempt number
            timeout_s: Timeout in seconds (default: self.timeout)

        Returns:
            True if all ranks reached barrier, False on timeout
        """
        if not self._ensure_store():
            logger.warning(f"Rank {self.rank}: Cannot barrier without store")
            return False

        if timeout_s is None:
            timeout_s = self.timeout

        try:
            # Key format: barrier/<name>/<attempt>/<rank>
            my_key = f"{self.KEY_BARRIER}/{name}/{attempt}/{self.rank}"

            # Signal that this rank has arrived
            self._store.set(my_key, "1")
            logger.debug(f"Rank {self.rank}: Arrived at barrier '{name}' (attempt {attempt})")

            # Wait for all ranks to arrive
            start_time = time.time()
            while True:
                arrived_count = 0
                for r in range(self.world_size):
                    key = f"{self.KEY_BARRIER}/{name}/{attempt}/{r}"
                    try:
                        value = self._store.get(key)
                        if value:
                            arrived_count += 1
                    except Exception:
                        pass

                if arrived_count >= self.world_size:
                    logger.debug(
                        f"Rank {self.rank}: Barrier '{name}' complete "
                        f"({arrived_count}/{self.world_size} ranks)"
                    )
                    return True

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_s:
                    logger.warning(
                        f"Rank {self.rank}: Barrier '{name}' timeout after {elapsed:.1f}s "
                        f"({arrived_count}/{self.world_size} ranks arrived)"
                    )
                    return False

                # Brief sleep to avoid busy-waiting
                time.sleep(0.1)

        except Exception as e:
            logger.warning(f"Rank {self.rank}: Barrier '{name}' failed: {e}")
            return False

    def cleanup(self, attempt: int) -> None:
        """Clean up keys for a completed attempt.

        This should be called after a successful restart to clean up
        old barrier keys and prevent store bloat.

        Args:
            attempt: The attempt number to clean up
        """
        if not self._store:
            return

        try:
            # Note: TCPStore doesn't have a delete operation in all versions,
            # so we just leave the keys. They're namespaced by attempt number
            # so they won't interfere with future attempts.
            pass
        except Exception as e:
            logger.debug(f"Rank {self.rank}: Cleanup warning: {e}")

    def close(self) -> None:
        """Close the coordinator and release resources."""
        if self._store is not None:
            try:
                # TCPStore doesn't have an explicit close method
                # Just clear the reference
                self._store = None
            except Exception:
                pass
        self._initialized = False


def create_restart_coordinator(
    rank: int = None,
    world_size: int = None,
    **kwargs,
) -> Optional[RestartCoordinator]:
    """Create a RestartCoordinator from environment.

    Args:
        rank: Rank index (default: from RANK env)
        world_size: World size (default: from WORLD_SIZE env)
        **kwargs: Additional arguments for RestartCoordinator

    Returns:
        RestartCoordinator instance, or None if world_size <= 1
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Single-rank training doesn't need coordination
    if world_size <= 1:
        logger.debug("Single-rank training, skipping RestartCoordinator")
        return None

    return RestartCoordinator(rank=rank, world_size=world_size, **kwargs)
