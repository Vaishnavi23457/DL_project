import numpy as np
import pytest

from bml.metrics import (accuracy, classification_report, confusion_matrix,
                          expected_calibration_error, format_confusion_md,
                          macro_f1, per_class_accuracy)


def test_accuracy_basic():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert accuracy([], []) == 0.0


def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0].tolist() == [1, 1, 0]
    assert cm[1].tolist() == [0, 2, 0]
    assert cm[2].tolist() == [1, 0, 0]
    assert cm.sum() == 5


def test_per_class_accuracy_and_macro_f1():
    cm = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 0])
    pca = per_class_accuracy(cm)
    assert pca[0] == 1.0
    assert pca[1] == 0.5
    assert 0.0 <= macro_f1(cm) <= 1.0
    # perfect predictions -> macro F1 = 1
    assert macro_f1(confusion_matrix([0, 1, 2], [0, 1, 2])) == pytest.approx(1.0)


def test_empty_class_row_is_zero_not_nan():
    cm = confusion_matrix([0, 0], [0, 1], num_classes=3)
    pca = per_class_accuracy(cm)
    assert pca[2] == 0.0 and not np.isnan(pca).any()


def test_ece_perfectly_calibrated_is_zero():
    y = [0, 1, 0, 1]
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_accepts_logits():
    y = [0, 1]
    logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
    assert expected_calibration_error(y, logits) == pytest.approx(0.0, abs=1e-6)


def test_classification_report_and_markdown():
    rep = classification_report([0, 1], [0, 1], class_names=["cat", "dog"])
    assert rep["accuracy"] == 1.0
    assert rep["per_class_accuracy"] == {"cat": 1.0, "dog": 1.0}
    md = format_confusion_md(np.array(rep["confusion_matrix"]), class_names=["cat", "dog"])
    assert md.startswith("| true \\ pred |")
    assert "**cat**" in md
