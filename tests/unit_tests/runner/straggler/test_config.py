from flagscale.runner.straggler.config import StragglerConfig


def test_straggler_config_defaults():
    config = StragglerConfig()

    assert config.enabled is True
    assert config.profiling_interval == 10
    assert config.report_interval_steps == 100
    assert config.straggler_threshold == 1.5
    assert "forward_backward" in config.monitor_sections
    assert "optimizer" in config.monitor_sections


def test_straggler_config_normalizes_values():
    config = StragglerConfig(
        profiling_interval=0,
        report_interval_steps=0,
        sample_size=0,
        warmup_steps=-1,
        max_stragglers_to_report=0,
    )

    assert config.profiling_interval == 1
    assert config.report_interval_steps == 1
    assert config.sample_size == 1
    assert config.warmup_steps == 0
    assert config.max_stragglers_to_report == 1
