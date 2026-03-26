from flagscale.runner.straggler.report import StragglerReport


def test_report_to_dict():
    report = StragglerReport(
        step=5,
        section_scores={"forward_backward": {0: 0.1, 1: 0.2}},
        gpu_scores={0: 10.0, 1: 5.0},
        straggler_ranks=[1],
        node_names={0: "node0", 1: "node1"},
        timestamp=123.0,
    )

    report_dict = report.to_dict()

    assert report_dict["step"] == 5
    assert report_dict["straggler_ranks"] == [1]
    assert report_dict["node_names"][1] == "node1"


def test_report_to_text_contains_summary():
    report = StragglerReport(
        step=8,
        section_scores={"optimizer": {0: 0.01, 1: 0.03}},
        straggler_ranks=[1],
        node_names={0: "node0", 1: "node1"},
    )

    text = report.to_text()

    assert "step 8" in text.lower()
    assert "Detected stragglers" in text
    assert "optimizer" in text
    assert "node1" in text
