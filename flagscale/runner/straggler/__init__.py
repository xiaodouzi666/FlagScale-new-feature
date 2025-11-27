"""
Straggler detection module for FlagScale.

This module provides infrastructure for detecting and reporting performance
stragglers in distributed training scenarios.
"""

from .config import StragglerConfig
from .detector import StragglerDetector
from .report import StragglerReport
from .section import SectionContext, OptionalSectionContext, create_section_decorator, SectionProfiler
from .comm import CommStatsCollector, CommProfiler, NCCLCommHook, GlooCommHook
from .healthcheck import NetworkHealthChecker, ElasticTrainingHealthChecker

__all__ = [
    "StragglerConfig",
    "StragglerDetector",
    "StragglerReport",
    "SectionContext",
    "OptionalSectionContext",
    "create_section_decorator",
    "SectionProfiler",
    "CommStatsCollector",
    "CommProfiler",
    "NCCLCommHook",
    "GlooCommHook",
    "NetworkHealthChecker",
    "ElasticTrainingHealthChecker",
]

__version__ = "0.1.0"
