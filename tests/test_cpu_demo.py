from psbd_nlp.cpu_demo import evaluate_psbd_scores, run_cpu_psbd_demo
from psbd_nlp.data import make_synthetic_backdoor_samples


def test_cpu_demo_flags_some_samples():
    rows = run_cpu_psbd_demo(make_synthetic_backdoor_samples())

    assert rows
    assert "shift_score" in rows[0]
    assert "is_suspicious" in rows[0]
    assert sum(int(bool(row["is_suspicious"])) for row in rows) >= 1


def test_cpu_demo_evaluation_returns_core_metrics():
    rows = run_cpu_psbd_demo(make_synthetic_backdoor_samples())
    metrics = evaluate_psbd_scores(rows)

    for key in ("accuracy", "precision", "recall", "f1", "auroc"):
        assert key in metrics
