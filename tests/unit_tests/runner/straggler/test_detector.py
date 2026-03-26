from flagscale.runner.straggler.config import StragglerConfig
from flagscale.runner.straggler.detector import StragglerDetector


def test_detector_profile_and_report_schedule():
    detector = StragglerDetector(
        StragglerConfig(profiling_interval=2, report_interval_steps=3, warmup_steps=1)
    )

    assert detector.should_profile(step=0) is False
    assert detector.should_profile(step=1) is True
    assert detector.should_profile(step=2) is False
    assert detector.should_profile(step=3) is True
    assert detector.should_report(step=2) is False
    assert detector.should_report(step=3) is True


def test_detector_identifies_stragglers_from_times():
    detector = StragglerDetector(StragglerConfig(straggler_threshold=1.5))

    section_times = {
        "forward_backward": {0: 0.10, 1: 0.11, 2: 0.20},
        "optimizer": {0: 0.01, 1: 0.01, 2: 0.03},
    }

    stragglers = detector._identify_stragglers_from_times(section_times)

    assert stragglers == [2]


def test_detector_generate_report_uses_gathered_data(monkeypatch):
    detector = StragglerDetector(StragglerConfig(straggler_threshold=1.5), rank=0, world_size=2)

    monkeypatch.setattr(
        detector,
        "_gather_section_times_across_ranks",
        lambda: {
            "forward_backward": {0: 0.10, 1: 0.18},
            "optimizer": {0: 0.01, 1: 0.03},
        },
    )
    monkeypatch.setattr(detector, "_gather_node_names_across_ranks", lambda: {0: "node0", 1: "node1"})

    report = detector.generate_report(step=12)

    assert report.step == 12
    assert report.straggler_ranks == [1]
    assert report.node_names[1] == "node1"
    assert "forward_backward" in report.section_scores
