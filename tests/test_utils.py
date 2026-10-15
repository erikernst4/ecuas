"""Unit tests for metric lookup by id (ecuas.utils.get_metric_from_id)."""

import pytest
import torch

from ecuas.classification import ClassificationECE
from ecuas.confidences import ExpectedCalibrationError
from ecuas.constants import METRICS
from ecuas.utils import get_metric_from_id


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────


@pytest.fixture
def cls_inputs():
    """Logits and labels for a 3-class problem (200 samples).

    Confident predictions are correct and unconfident ones are random, so the
    model is under-confident in some bins and over-confident in others (with
    one-sided miscalibration, ECE would not depend on the number of bins).
    """
    torch.manual_seed(0)
    logits = torch.randn(200, 3) * 3
    probs, preds = torch.softmax(logits, dim=1).max(dim=1)
    labels = torch.where(probs > 0.8, preds, torch.randint(0, 3, (200,)))
    return logits, labels


@pytest.fixture
def conf_inputs():
    """Random confidences and correctness indicators (200 samples)."""
    torch.manual_seed(0)
    confidences = torch.rand(200)
    correctness = torch.rand(200) < confidences
    return confidences, correctness


def _inputs_for(id_name, cls_inputs, conf_inputs):
    return cls_inputs if id_name.startswith("cls_") else conf_inputs


def _compute_obj(metric, inputs):
    metric.update(*inputs)
    return metric.compute().item()


NORMALIZED_CONF_IDS = {"conf_norm_brier", "conf_norm_cross_entropy"}

PARAMETRIZED_IDS = [
    "conf_n-ecuas_n=3",
    "conf_gamma-ecuas_gamma=0.5",
    "conf_ece",
    "conf_ece_nbins=5",
    "cls_n-ecuas_n=3",
    "cls_norm_n-ecuas_n=3",
    "cls_gamma-ecuas_gamma=0.5",
    "cls_norm_gamma-ecuas_gamma=0.5",
    "cls_ece",
    "cls_ece_nbins=5",
]


# ──────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────


class TestGetMetricFromId:
    @pytest.mark.parametrize(
        "id_name",
        [
            pytest.param(
                k,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="obj is built as cls() and drops normalize=True",
                ),
            )
            if k in NORMALIZED_CONF_IDS
            else k
            for k in sorted(METRICS)
        ],
    )
    def test_registered_ids(self, id_name, cls_inputs, conf_inputs):
        """Every registered id's function and obj run and agree."""
        metric = get_metric_from_id(id_name)
        inputs = _inputs_for(id_name, cls_inputs, conf_inputs)
        assert metric["function"](*inputs) == pytest.approx(
            _compute_obj(metric["obj"], inputs)
        )

    @pytest.mark.parametrize("id_name", PARAMETRIZED_IDS)
    def test_parametrized_ids(self, id_name, cls_inputs, conf_inputs):
        """Parametrized ids build a function and obj that run and agree."""
        metric = get_metric_from_id(id_name)
        inputs = _inputs_for(id_name, cls_inputs, conf_inputs)
        assert metric["function"](*inputs) == pytest.approx(
            _compute_obj(metric["obj"], inputs)
        )

    @pytest.mark.parametrize(
        "prefix, metric_cls",
        [("conf_ece", ExpectedCalibrationError), ("cls_ece", ClassificationECE)],
    )
    def test_ece_nbins_is_applied(self, prefix, metric_cls, cls_inputs, conf_inputs):
        """The nbins in the id reaches both the function and the obj (issue #1)."""
        inputs = _inputs_for(prefix, cls_inputs, conf_inputs)
        expected = _compute_obj(metric_cls(n_bins=1), inputs)
        assert expected != pytest.approx(_compute_obj(metric_cls(n_bins=10), inputs))

        metric = get_metric_from_id(f"{prefix}_nbins=1")
        assert metric["function"](*inputs) == pytest.approx(expected)
        assert _compute_obj(metric["obj"], inputs) == pytest.approx(expected)

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Metric not found"):
            get_metric_from_id("not_a_metric")
