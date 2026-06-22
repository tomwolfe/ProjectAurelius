from certifier.optimizer import KernelOptimizer


def test_optimizer_returns_kernel() -> None:
    opt = KernelOptimizer()
    data = [("CCO", -1.5), ("CC=O", -2.1), ("C#N", -3.0)]
    kernel = opt.optimize(data)
    assert "version" in kernel
    assert "tom_parameters" in kernel
    assert "validation_metrics" in kernel
    assert kernel["version"] == "1.0.0"


def test_optimizer_accepts_domain_boundary() -> None:
    opt = KernelOptimizer()
    data = [("CCO", -1.5)]
    boundary = {"domain": "carbonate", "max_molecular_weight": 300}
    kernel = opt.optimize(data, domain_boundary=boundary)
    assert kernel["domain_boundary"]["domain"] == "carbonate"


def test_optimizer_empty_data() -> None:
    opt = KernelOptimizer()
    kernel = opt.optimize([])
    assert kernel is not None


def test_optimizer_validation_metrics() -> None:
    opt = KernelOptimizer()
    data = [("CCO", -1.0), ("CC=O", -2.0)]
    kernel = opt.optimize(data)
    metrics = kernel["validation_metrics"]
    assert "spearman_rho" in metrics
    assert "mae" in metrics
