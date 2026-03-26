"""Section timing helpers for FlagScale straggler detection."""

import time
from typing import Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SectionContext:
    """Context manager that records a timed training section."""

    def __init__(self, detector, name: str, profile_gpu: bool = False) -> None:
        self.detector = detector
        self.name = name
        self.profile_gpu = profile_gpu
        self.start_time: Optional[float] = None
        self.cuda_start_event = None
        self.cuda_end_event = None

    def __enter__(self):
        self.start_time = time.perf_counter()

        if self.profile_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        cpu_elapsed = end_time - self.start_time
        gpu_elapsed = None

        if self.cuda_start_event is not None and self.cuda_end_event is not None:
            torch.cuda.synchronize()
            self.cuda_end_event.record()
            torch.cuda.synchronize()
            gpu_elapsed = self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000.0

        if hasattr(self.detector, "record_section"):
            self.detector.record_section(self.name, cpu_elapsed, gpu_elapsed)

        return False


class OptionalSectionContext:
    """A no-op wrapper when profiling is disabled."""

    def __init__(self, detector, name: str, enabled: bool = True, profile_gpu: bool = False):
        self.enabled = enabled
        self.context = SectionContext(detector, name, profile_gpu=profile_gpu) if enabled else None

    def __enter__(self):
        if self.context is not None:
            return self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            return self.context.__exit__(exc_type, exc_val, exc_tb)
        return False
