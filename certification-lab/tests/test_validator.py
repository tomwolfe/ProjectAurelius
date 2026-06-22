from certifier.validator import UncertaintyAuditor


def test_auditor_returns_report() -> None:
    auditor = UncertaintyAuditor()
    kernel = {"version": "1.0.0", "tom_parameters": {}}
    data = [("CCO", -1.5), ("CC=O", -2.1)]
    report = auditor.audit(kernel, data)
    assert "coverage_probability" in report
    assert "pass" in report
    assert "n_samples" in report


def test_auditor_empty_data() -> None:
    auditor = UncertaintyAuditor()
    kernel = {"version": "1.0.0"}
    report = auditor.audit(kernel, [])
    assert report["n_samples"] == 0
    assert report["pass"] is False


def test_auditor_custom_confidence() -> None:
    auditor = UncertaintyAuditor(confidence_level=0.99)
    assert auditor.confidence_level == 0.99
