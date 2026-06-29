from src.eval import compute_classification_metrics, compute_roc_auc


def test_perfect_predictions_score_one():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_all_wrong_predictions_score_zero():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0


def test_roc_auc_perfect_separation_is_one():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert compute_roc_auc(y_true, scores) == 1.0
