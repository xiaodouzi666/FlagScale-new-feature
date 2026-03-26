"""Section-based straggler detector for distributed training."""

from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.distributed as dist

    TORCH_DIST_AVAILABLE = True
except ImportError:
    torch = None
    dist = None
    TORCH_DIST_AVAILABLE = False

from .config import StragglerConfig
from .report import StragglerReport


class StragglerDetector:
    """Collect timings and detect slow ranks."""

    def __init__(
        self,
        config: StragglerConfig,
        rank: int = 0,
        world_size: int = 1,
        node_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.node_name = node_name or config.node_name or f"rank-{rank}"
        self.section_timings: Dict[str, List[Tuple[int, float, Optional[float]]]] = defaultdict(list)
        self.current_step = 0
        self.enabled = config.enabled

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def reset(self) -> None:
        self.section_timings.clear()
        self.current_step = 0

    def increment_step(self) -> None:
        self.current_step += 1

    def should_profile(self, step: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if step is None:
            step = self.current_step
        if step < self.config.warmup_steps:
            return False
        return (step - self.config.warmup_steps) % self.config.profiling_interval == 0

    def should_report(self, step: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if step is None:
            step = self.current_step
        return step > 0 and step % self.config.report_interval_steps == 0

    def record_section(
        self,
        name: str,
        cpu_time: float,
        gpu_time: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled or name not in self.config.monitor_sections:
            return
        if step is None:
            step = self.current_step
        self.section_timings[name].append((step, cpu_time, gpu_time))

    def get_recent_section_time(self, section_name: str, num_samples: Optional[int] = None) -> Optional[float]:
        timings = self.section_timings.get(section_name, [])
        if not timings:
            return None

        if num_samples is None:
            num_samples = self.config.sample_size

        recent_timings = timings[-num_samples:]
        total = 0.0
        for _, cpu_time, gpu_time in recent_timings:
            total += gpu_time if gpu_time is not None else cpu_time
        return total / len(recent_timings)

    def _get_collective_device(self):
        if not TORCH_DIST_AVAILABLE or not dist.is_initialized():
            if torch is not None and torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu") if torch is not None else "cpu"

        backend = str(dist.get_backend()).lower()
        if "gloo" in backend and "nccl" not in backend and "cuda:" not in backend and "flagcx" not in backend:
            return torch.device("cpu")
        if torch is not None and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _gather_section_times_across_ranks(self) -> Dict[str, Dict[int, float]]:
        result: Dict[str, Dict[int, float]] = {}

        if not TORCH_DIST_AVAILABLE or not dist.is_initialized():
            for section_name in self.config.monitor_sections:
                avg_time = self.get_recent_section_time(section_name)
                if avg_time is not None:
                    result[section_name] = {self.rank: avg_time}
            return result

        device = self._get_collective_device()
        for section_name in self.config.monitor_sections:
            avg_time = self.get_recent_section_time(section_name)
            local_time = -1.0 if avg_time is None else avg_time
            local_tensor = torch.tensor([local_time], dtype=torch.float64, device=device)
            gathered = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_tensor)

            rank_times = {}
            for rank, tensor in enumerate(gathered):
                time_value = tensor.item()
                if time_value >= 0.0:
                    rank_times[rank] = time_value

            if rank_times:
                result[section_name] = rank_times

        return result

    def _gather_node_names_across_ranks(self) -> Dict[int, str]:
        if not TORCH_DIST_AVAILABLE or not dist.is_initialized():
            return {self.rank: self.node_name}

        node_names = [None] * self.world_size
        dist.all_gather_object(node_names, self.node_name)
        return {rank: name for rank, name in enumerate(node_names) if name is not None}

    def _identify_stragglers_from_times(
        self,
        section_times: Dict[str, Dict[int, float]],
        threshold: Optional[float] = None,
    ) -> List[int]:
        if threshold is None:
            threshold = self.config.straggler_threshold
        if not section_times:
            return []

        total_times = defaultdict(float)
        for rank_times in section_times.values():
            for rank, time_value in rank_times.items():
                total_times[rank] += time_value

        if not total_times:
            return []

        fastest_rank, fastest_time = min(total_times.items(), key=lambda item: item[1])
        if fastest_time <= 0:
            return []

        stragglers = []
        for rank, total_time in total_times.items():
            if rank == fastest_rank:
                continue
            slowdown = total_time / fastest_time
            if slowdown >= threshold:
                stragglers.append(rank)

        return sorted(stragglers)[: self.config.max_stragglers_to_report]

    def generate_report(self, step: Optional[int] = None) -> StragglerReport:
        if step is None:
            step = self.current_step

        section_scores = self._gather_section_times_across_ranks()
        node_names = self._gather_node_names_across_ranks()
        straggler_ranks = self._identify_stragglers_from_times(section_scores)

        gpu_scores = {}
        ranks = sorted({rank for section in section_scores.values() for rank in section})
        for rank in ranks:
            total_time = sum(section.get(rank, 0.0) for section in section_scores.values())
            if total_time > 0:
                gpu_scores[rank] = 1.0 / total_time

        return StragglerReport(
            step=step,
            section_scores=section_scores,
            gpu_scores=gpu_scores,
            straggler_ranks=straggler_ranks,
            node_names=node_names,
            timestamp=time.time(),
        )
