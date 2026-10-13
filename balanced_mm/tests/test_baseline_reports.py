"""Independent metric checks for the new comparison's detailed class reports."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_baseline_comparison import extended_metrics


def test_imbalanced_confusion_preserves_macro_weighted_and_micro_distinction():
    report = extended_metrics(np.array([[9, 1], [2, 0]]), ["majority", "minority"])
    assert report["accuracy"] == pytest.approx(9 / 12)
    assert report["macro_precision"] == pytest.approx((9 / 11) / 2)
    assert report["macro_recall"] == pytest.approx(0.45)
    assert report["macro_f1"] == pytest.approx((18 / 21) / 2)
    assert report["weighted_f1"] == pytest.approx((18 / 21) * (10 / 12))
    assert report["micro_f1"] == report["accuracy"]
    assert report["balanced_accuracy"] == report["macro_recall"]
    assert [c["support"] for c in report["classes"]] == [10, 2]
    assert report["classes"][1]["precision"] == report["classes"][1]["f1"] == 0


def test_unpredicted_class_has_finite_zero_precision_and_class_name_is_preserved():
    report = extended_metrics(np.array([[2, 0, 0], [1, 0, 0], [0, 0, 3]]), ["a", "b", "c"])
    assert report["classes"][1] == {"class": "b", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert report["test_n"] == 6
