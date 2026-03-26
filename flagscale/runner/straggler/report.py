"""Report structures for FlagScale straggler detection."""

from typing import Any, Dict, List, Optional


class StragglerReport:
    """Serializable straggler report."""

    def __init__(
        self,
        step: int,
        section_scores: Optional[Dict[str, Dict[int, float]]] = None,
        gpu_scores: Optional[Dict[int, float]] = None,
        straggler_ranks: Optional[List[int]] = None,
        node_names: Optional[Dict[int, str]] = None,
        comm_stats: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.step = step
        self.section_scores = section_scores or {}
        self.gpu_scores = gpu_scores or {}
        self.straggler_ranks = straggler_ranks or []
        self.node_names = node_names or {}
        self.comm_stats = comm_stats or {}
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "section_scores": self.section_scores,
            "gpu_scores": self.gpu_scores,
            "straggler_ranks": self.straggler_ranks,
            "node_names": self.node_names,
            "comm_stats": self.comm_stats,
            "timestamp": self.timestamp,
        }

    def to_text(self) -> str:
        lines = [f"=== Straggler Report at step {self.step} ==="]

        if self.straggler_ranks:
            lines.append("Detected stragglers:")
            for rank in self.straggler_ranks:
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  rank {rank} ({node_name})")
        else:
            lines.append("No stragglers detected.")

        if self.section_scores:
            lines.append("")
            lines.append("Section timings (ms):")
            for section_name, rank_times in sorted(self.section_scores.items()):
                if not rank_times:
                    continue
                times_ms = {rank: value * 1000.0 for rank, value in rank_times.items()}
                min_time = min(times_ms.values())
                max_time = max(times_ms.values())
                avg_time = sum(times_ms.values()) / len(times_ms)
                lines.append(
                    f"  {section_name}: min={min_time:.2f}, max={max_time:.2f}, avg={avg_time:.2f}"
                )
                for rank, value in sorted(times_ms.items()):
                    node_name = self.node_names.get(rank, f"rank-{rank}")
                    lines.append(f"    rank {rank} ({node_name}): {value:.2f}")

        if self.gpu_scores:
            lines.append("")
            lines.append("GPU scores (higher is faster):")
            for rank, value in sorted(self.gpu_scores.items()):
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  rank {rank} ({node_name}): {value:.6f}")

        return "\n".join(lines)
