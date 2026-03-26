"""Configuration objects for FlagScale straggler detection."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class StragglerConfig:
    """Configuration for section-based straggler detection."""

    enabled: bool = True
    scores_to_compute: Literal["relative", "individual", "all"] = "all"
    gather_on_rank0: bool = True
    profiling_interval: int = 10
    report_interval_steps: int = 100
    node_name: Optional[str] = None
    monitor_sections: List[str] = field(
        default_factory=lambda: ["dataloader", "forward", "backward", "optimizer", "forward_backward"]
    )
    enable_comm_logging: bool = False
    enable_gpu_profile: bool = True
    straggler_threshold: float = 1.5
    max_stragglers_to_report: int = 5
    sample_size: int = 5
    warmup_steps: int = 10

    def __post_init__(self) -> None:
        self.profiling_interval = max(1, int(self.profiling_interval))
        self.report_interval_steps = max(1, int(self.report_interval_steps))
        self.sample_size = max(1, int(self.sample_size))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.max_stragglers_to_report = max(1, int(self.max_stragglers_to_report))
        if self.straggler_threshold < 1.0:
            raise ValueError("straggler_threshold must be >= 1.0")
