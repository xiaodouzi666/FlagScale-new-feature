import unittest
from flagscale.runner.in_process.rank_assignment import RankAssignmentCtx, MaxActiveWorldSize, SimpleRankAssignment
from flagscale.runner.in_process.state import RankState, RankMode

class TestRankAssignment(unittest.TestCase):
    def test_simple_assignment(self):
        # Scenario: 4 ranks, rank 1 active
        state = RankState(initial_rank=1, world_size=4, rank=1)
        ctx = RankAssignmentCtx(state=state, terminated_ranks=set())
        
        strategy = SimpleRankAssignment()
        ctx = strategy(ctx)
        
        self.assertEqual(ctx.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx.state.rank, 1)
        
        # Scenario: Rank 1 terminated
        ctx.terminated_ranks.add(1)
        ctx = strategy(ctx)
        self.assertEqual(ctx.state.mode, RankMode.TERMINATED)

    def test_max_active_world_size_basic(self):
        # Scenario: 4 ranks, max active 2
        # Expected:
        # Rank 0 -> Logical 0 (Active)
        # Rank 1 -> Logical 1 (Active)
        # Rank 2 -> Inactive
        # Rank 3 -> Inactive
        
        strategy = MaxActiveWorldSize(max_active_world_size=2)
        
        # Check Rank 0
        state0 = RankState(initial_rank=0, world_size=4)
        ctx0 = strategy(RankAssignmentCtx(state0, set()))
        self.assertEqual(ctx0.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx0.state.rank, 0)
        self.assertEqual(ctx0.state.active_world_size, 2)
        
        # Check Rank 1
        state1 = RankState(initial_rank=1, world_size=4)
        ctx1 = strategy(RankAssignmentCtx(state1, set()))
        self.assertEqual(ctx1.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx1.state.rank, 1)
        
        # Check Rank 2
        state2 = RankState(initial_rank=2, world_size=4)
        ctx2 = strategy(RankAssignmentCtx(state2, set()))
        self.assertEqual(ctx2.state.mode, RankMode.INACTIVE)
        self.assertEqual(ctx2.state.rank, -1)
        
    def test_max_active_world_size_replacement(self):
        # Scenario: 4 ranks, max active 2. Rank 0 fails.
        # Expected:
        # Rank 0 -> Terminated
        # Rank 1 -> Logical 0 (Shifted)
        # Rank 2 -> Logical 1 (Shifted)
        # Rank 3 -> Inactive
        
        strategy = MaxActiveWorldSize(max_active_world_size=2)
        terminated = {0}
        
        # Check Rank 0 (Failed)
        state0 = RankState(initial_rank=0, world_size=4)
        ctx0 = strategy(RankAssignmentCtx(state0, terminated))
        self.assertEqual(ctx0.state.mode, RankMode.TERMINATED)
        
        # Check Rank 1 (Now Logical 0)
        state1 = RankState(initial_rank=1, world_size=4)
        ctx1 = strategy(RankAssignmentCtx(state1, terminated))
        self.assertEqual(ctx1.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx1.state.rank, 0)
        self.assertEqual(ctx1.state.active_world_size, 2)
        
        # Check Rank 2 (Now Logical 1)
        state2 = RankState(initial_rank=2, world_size=4)
        ctx2 = strategy(RankAssignmentCtx(state2, terminated))
        self.assertEqual(ctx2.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx2.state.rank, 1)
        
        # Check Rank 3 (Still Inactive)
        state3 = RankState(initial_rank=3, world_size=4)
        ctx3 = strategy(RankAssignmentCtx(state3, terminated))
        self.assertEqual(ctx3.state.mode, RankMode.INACTIVE)

    def test_max_active_world_size_double_failure(self):
        # Scenario: 4 ranks, max active 2. Rank 0 and 2 fail.
        # Expected:
        # Rank 0 -> Terminated
        # Rank 1 -> Logical 0
        # Rank 2 -> Terminated
        # Rank 3 -> Logical 1 (Activated from spare)
        
        strategy = MaxActiveWorldSize(max_active_world_size=2)
        terminated = {0, 2}
        
        # Rank 1
        state1 = RankState(initial_rank=1, world_size=4)
        ctx1 = strategy(RankAssignmentCtx(state1, terminated))
        self.assertEqual(ctx1.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx1.state.rank, 0)
        
        # Rank 3
        state3 = RankState(initial_rank=3, world_size=4)
        ctx3 = strategy(RankAssignmentCtx(state3, terminated))
        self.assertEqual(ctx3.state.mode, RankMode.ACTIVE)
        self.assertEqual(ctx3.state.rank, 1)

if __name__ == '__main__':
    unittest.main()
