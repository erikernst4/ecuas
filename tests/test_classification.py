"""Unit tests for multiclass classification metrics."""

import pytest
import torch
import numpy as np

from ecuas.classification import (
    ClassificationErrorRate,
    ClassificationCrossEntropy,
    ClassificationBrierScore,
    ClassificationAUC,
    ClassificationECE,
    ClassificationECUAS,
    ClassificationLogLog,
    ClassificationGammaECUAS,
    ClassificationAURC,
    ClassificationFPR95,
)


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────


@pytest.fixture
def perfect_3class():
    """Logits that perfectly predict labels for 3 classes (30 samples)."""
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    logits = torch.zeros(30, 3)
    for i, label in enumerate(labels):
        logits[i, label] = 10.0  # dominant logit
    return logits, labels


@pytest.fixture
def uniform_3class():
    """Uniform (flat) logits — maximally uncertain — for 3 balanced classes."""
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    logits = torch.zeros(30, 3)  # all equal → softmax = 1/3 each
    return logits, labels


@pytest.fixture
def random_5class():
    """Random logits for 5 classes (100 samples) with a fixed seed."""
    torch.manual_seed(0)
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    return logits, labels


# ──────────────────────────────────────────────────────
# ClassificationErrorRate
# ──────────────────────────────────────────────────────


class TestClassificationErrorRate:
    def test_perfect_predictions_unnormalized(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationErrorRate(normalize=False)
        m.update(logits, labels)
        assert m.compute().item() == 0.0

    def test_perfect_predictions_normalized(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationErrorRate(normalize=True)
        m.update(logits, labels)
        assert m.compute().item() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_logits_unnormalized(self, uniform_3class):
        """Uniform logits → argmax picks class 0 for every sample."""
        logits, labels = uniform_3class
        m = ClassificationErrorRate(normalize=False)
        m.update(logits, labels)
        er = m.compute().item()
        # Classes 1 and 2 are always wrong → 20/30 ≈ 0.667
        assert er == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_normalized_equals_one_for_uniform(self, uniform_3class):
        """Uniform logits should give normalized ER ≈ 1.0 (same as prior)."""
        logits, labels = uniform_3class
        m = ClassificationErrorRate(normalize=True)
        m.update(logits, labels)
        er = m.compute().item()
        assert er == pytest.approx(1.0, abs=1e-4)

    def test_nan_logits_raises(self, perfect_3class):
        logits, labels = perfect_3class
        logits[0, 0] = float("nan")
        m = ClassificationErrorRate()
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, labels)

    def test_nan_labels_raises(self, perfect_3class):
        logits, labels = perfect_3class
        labels_f = labels.float()
        labels_f[0] = float("nan")
        m = ClassificationErrorRate()
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, labels_f)

    def test_wrong_dimensions_raises(self):
        m = ClassificationErrorRate()
        with pytest.raises(ValueError, match="Logits must be 2D"):
            m.update(torch.randn(10), torch.randint(0, 2, (10,)))

    def test_empty_raises(self):
        m = ClassificationErrorRate()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_incremental_update(self, random_5class):
        logits, labels = random_5class
        m1 = ClassificationErrorRate(normalize=False)
        m1.update(logits, labels)
        v1 = m1.compute().item()

        m2 = ClassificationErrorRate(normalize=False)
        m2.update(logits[:50], labels[:50])
        m2.update(logits[50:], labels[50:])
        v2 = m2.compute().item()
        assert v1 == pytest.approx(v2, abs=1e-6)

    def test_reduction_sum(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationErrorRate(normalize=False, reduction="sum")
        m.update(logits, labels)
        assert m.compute().item() == 0.0

    def test_reduction_none(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationErrorRate(normalize=False, reduction="none")
        m.update(logits, labels)
        result = m.compute()
        assert result.shape == (30,)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationErrorRate.create_shortcut_function(normalize=False)
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationCrossEntropy
# ──────────────────────────────────────────────────────


class TestClassificationCrossEntropy:
    def test_perfect_predictions(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationCrossEntropy(normalize=False)
        m.update(logits, labels)
        ce = m.compute().item()
        assert ce < 0.01  # near-zero CE

    def test_normalized_perfect(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationCrossEntropy(normalize=True)
        m.update(logits, labels)
        ce = m.compute().item()
        assert ce < 0.01

    def test_uniform_logits_unnormalized(self, uniform_3class):
        logits, labels = uniform_3class
        m = ClassificationCrossEntropy(normalize=False)
        m.update(logits, labels)
        ce = m.compute().item()
        # Uniform → CE = log(K) ≈ 1.099
        assert ce == pytest.approx(np.log(3), abs=0.01)

    def test_normalized_uniform_near_one(self, uniform_3class):
        """Uniform logits on balanced classes → normalized CE ≈ 1.0."""
        logits, labels = uniform_3class
        m = ClassificationCrossEntropy(normalize=True)
        m.update(logits, labels)
        ce = m.compute().item()
        assert ce == pytest.approx(1.0, abs=0.05)

    def test_nan_raises(self):
        m = ClassificationCrossEntropy()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationCrossEntropy()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_incremental_update(self, random_5class):
        logits, labels = random_5class
        m1 = ClassificationCrossEntropy(normalize=False)
        m1.update(logits, labels)
        v1 = m1.compute().item()

        m2 = ClassificationCrossEntropy(normalize=False)
        m2.update(logits[:50], labels[:50])
        m2.update(logits[50:], labels[50:])
        v2 = m2.compute().item()
        assert v1 == pytest.approx(v2, abs=1e-5)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationCrossEntropy.create_shortcut_function(normalize=False)
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationBrierScore
# ──────────────────────────────────────────────────────


class TestClassificationBrierScore:
    def test_perfect_predictions(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationBrierScore(normalize=False)
        m.update(logits, labels)
        bs = m.compute().item()
        assert bs < 0.01

    def test_uniform_logits(self, uniform_3class):
        logits, labels = uniform_3class
        m = ClassificationBrierScore(normalize=False)
        m.update(logits, labels)
        bs = m.compute().item()
        # softmax = 1/3 each, one-hot: mean (1/3-1)^2 + 2*(1/3-0)^2 = 4/9+2/9 = 2/3 per sample → mean over dim=1 = 2/9
        expected = 2.0 / 9.0
        assert bs == pytest.approx(expected, abs=0.01)

    def test_normalized_perfect_near_zero(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationBrierScore(normalize=True)
        m.update(logits, labels)
        bs = m.compute().item()
        assert bs < 0.01

    def test_nan_raises(self):
        m = ClassificationBrierScore()
        logits = torch.randn(5, 3)
        logits[2, 1] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationBrierScore()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_incremental_update(self, random_5class):
        logits, labels = random_5class
        m1 = ClassificationBrierScore(normalize=False)
        m1.update(logits, labels)
        v1 = m1.compute().item()

        m2 = ClassificationBrierScore(normalize=False)
        m2.update(logits[:50], labels[:50])
        m2.update(logits[50:], labels[50:])
        v2 = m2.compute().item()
        assert v1 == pytest.approx(v2, abs=1e-5)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationBrierScore.create_shortcut_function(normalize=False)
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationAUC
# ──────────────────────────────────────────────────────


class TestClassificationAUC:
    def test_perfect_predictions(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationAUC()
        m.update(logits, labels)
        auc = m.compute().item()
        # All correct → single class in correctness → BinaryAUROC returns 0
        # (no negative examples to rank against)
        assert 0.0 <= auc <= 1.0

    def test_random_range(self, random_5class):
        logits, labels = random_5class
        m = ClassificationAUC()
        m.update(logits, labels)
        auc = m.compute().item()
        assert 0.0 <= auc <= 1.0

    def test_nan_raises(self):
        m = ClassificationAUC()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationAUC()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationAUC.create_shortcut_function()
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationECE
# ──────────────────────────────────────────────────────


class TestClassificationECE:
    def test_perfect_predictions_low_ece(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationECE(n_bins=10)
        m.update(logits, labels)
        ece = m.compute().item()
        assert ece < 0.05

    def test_random_range(self, random_5class):
        logits, labels = random_5class
        m = ClassificationECE(n_bins=10)
        m.update(logits, labels)
        ece = m.compute().item()
        assert 0.0 <= ece <= 1.0

    def test_nan_raises(self):
        m = ClassificationECE()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationECE()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationECE.create_shortcut_function()
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationECUAS
# ──────────────────────────────────────────────────────


class TestClassificationECUAS:
    def test_perfect_predictions_low_cost(self, perfect_3class):
        """Perfect logits → very high confidence → near-zero cost."""
        logits, labels = perfect_3class
        m = ClassificationECUAS(n=0, normalize=False)
        m.update(logits, labels)
        cost = m.compute().item()
        assert cost < 0.01

    def test_normalized_perfect_near_zero(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationECUAS(n=0, normalize=True)
        m.update(logits, labels)
        cost = m.compute().item()
        assert cost < 0.05

    def test_n_positive(self, random_5class):
        """n>0 variant should produce finite results."""
        logits, labels = random_5class
        for n in [1, 2, 5]:
            m = ClassificationECUAS(n=n, normalize=False)
            m.update(logits, labels)
            cost = m.compute().item()
            assert np.isfinite(cost), f"Non-finite cost for n={n}"

    def test_negative_n_raises(self):
        m = ClassificationECUAS(n=-1, normalize=False)
        m.update(torch.randn(10, 3), torch.randint(0, 3, (10,)))
        with pytest.raises(ValueError, match="n must be non-negative"):
            m.compute()

    def test_uniform_higher_cost(self, perfect_3class, uniform_3class):
        """Uniform logits should produce higher cost than perfect logits."""
        m_perf = ClassificationECUAS(n=0, normalize=False)
        m_perf.update(*perfect_3class)
        cost_perf = m_perf.compute().item()

        m_unif = ClassificationECUAS(n=0, normalize=False)
        m_unif.update(*uniform_3class)
        cost_unif = m_unif.compute().item()

        assert cost_unif > cost_perf

    def test_nan_raises(self):
        m = ClassificationECUAS()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationECUAS()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_incremental_update(self, random_5class):
        logits, labels = random_5class
        m1 = ClassificationECUAS(n=0, normalize=False)
        m1.update(logits, labels)
        v1 = m1.compute().item()

        m2 = ClassificationECUAS(n=0, normalize=False)
        m2.update(logits[:50], labels[:50])
        m2.update(logits[50:], labels[50:])
        v2 = m2.compute().item()
        assert v1 == pytest.approx(v2, abs=1e-5)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationECUAS.create_shortcut_function(normalize=False)
        val = fn(logits, labels, n=0)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationLogLog
# ──────────────────────────────────────────────────────


class TestClassificationLogLog:
    def test_perfect_predictions(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationLogLog(normalize=False)
        m.update(logits, labels)
        cost = m.compute().item()
        assert np.isfinite(cost)

    def test_normalized(self, random_5class):
        logits, labels = random_5class
        m = ClassificationLogLog(normalize=True)
        m.update(logits, labels)
        cost = m.compute().item()
        assert np.isfinite(cost)

    def test_unnormalized(self, random_5class):
        logits, labels = random_5class
        m = ClassificationLogLog(normalize=False)
        m.update(logits, labels)
        cost = m.compute().item()
        assert np.isfinite(cost)

    def test_nan_raises(self):
        m = ClassificationLogLog()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationLogLog()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_incremental_update(self, random_5class):
        logits, labels = random_5class
        m1 = ClassificationLogLog(normalize=False)
        m1.update(logits, labels)
        v1 = m1.compute().item()

        m2 = ClassificationLogLog(normalize=False)
        m2.update(logits[:50], labels[:50])
        m2.update(logits[50:], labels[50:])
        v2 = m2.compute().item()
        assert v1 == pytest.approx(v2, abs=1e-5)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationLogLog.create_shortcut_function(normalize=False)
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationGammaECUAS
# ──────────────────────────────────────────────────────


class TestClassificationGammaECUAS:
    def test_perfect_predictions_normalized(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationGammaECUAS(gamma=0.5, normalize=True)
        m.update(logits, labels)
        cost = m.compute().item()
        assert np.isfinite(cost)

    def test_unnormalized(self, random_5class):
        logits, labels = random_5class
        m = ClassificationGammaECUAS(gamma=0.5, normalize=False)
        m.update(logits, labels)
        cost = m.compute().item()
        assert 0.0 <= cost <= 1.0

    def test_different_gammas(self, random_5class):
        logits, labels = random_5class
        costs = []
        for gamma in [0.1, 0.5, 0.9]:
            m = ClassificationGammaECUAS(gamma=gamma, normalize=False)
            m.update(logits, labels)
            costs.append(m.compute().item())
        # Different gammas should produce different costs
        assert not all(abs(c - costs[0]) < 1e-6 for c in costs[1:])

    def test_all_abstain_at_low_gamma(self, perfect_3class):
        """With perfect predictions (high confidence), all should abstain when gamma is high."""
        logits, labels = perfect_3class
        m = ClassificationGammaECUAS(gamma=0.01, normalize=False)
        m.update(logits, labels)
        cost = m.compute().item()
        # All high confidence → all abstain → cost = gamma
        assert cost == pytest.approx(0.01, abs=1e-4)

    def test_nan_raises(self):
        m = ClassificationGammaECUAS()
        logits = torch.randn(5, 3)
        logits[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            m.update(logits, torch.tensor([0, 1, 2, 0, 1]))

    def test_empty_raises(self):
        m = ClassificationGammaECUAS()
        with pytest.raises(ValueError, match="No data"):
            m.compute()

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationGammaECUAS.create_shortcut_function(normalize=False)
        val = fn(logits, labels, gamma=0.5)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationAURC
# ──────────────────────────────────────────────────────


class TestClassificationAURC:
    def test_perfect_predictions(self, perfect_3class):
        logits, labels = perfect_3class
        m = ClassificationAURC()
        m.update(logits, labels)
        aurc = m.compute().item()
        assert aurc < 0.01

    def test_random_finite(self, random_5class):
        logits, labels = random_5class
        m = ClassificationAURC()
        m.update(logits, labels)
        aurc = m.compute().item()
        assert np.isfinite(aurc)

    def test_shortcut(self, random_5class):
        logits, labels = random_5class
        fn = ClassificationAURC.create_shortcut_function()
        val = fn(logits, labels)
        assert isinstance(val, float) and np.isfinite(val)


# ──────────────────────────────────────────────────────
# ClassificationFPR95
# ──────────────────────────────────────────────────────


class TestClassificationFPR95:
    def test_shortcut_exists(self):
        """ClassificationFPR95 has a create_shortcut_function classmethod."""
        fn = ClassificationFPR95.create_shortcut_function()
        assert callable(fn)


# ──────────────────────────────────────────────────────
# Cross-metric integration
# ──────────────────────────────────────────────────────


class TestClassificationIntegration:
    def test_all_metrics_on_same_data(self, random_5class):
        """All classification metrics should produce finite results on the same data."""
        logits, labels = random_5class
        metrics = {
            "er": ClassificationErrorRate(normalize=False),
            "ce": ClassificationCrossEntropy(normalize=False),
            "bs": ClassificationBrierScore(normalize=False),
            "auc": ClassificationAUC(),
            "ece": ClassificationECE(),
            "ecuas_0": ClassificationECUAS(n=0, normalize=False),
            "ecuas_1": ClassificationECUAS(n=1, normalize=False),
            "loglog": ClassificationLogLog(normalize=False),
            "gamma": ClassificationGammaECUAS(gamma=0.5, normalize=False),
            "aurc": ClassificationAURC(),
        }
        for m in metrics.values():
            m.update(logits, labels)

        for name, m in metrics.items():
            val = m.compute().item()
            assert np.isfinite(val), f"Metric {name} returned non-finite: {val}"

    def test_normalized_bounded(self, random_5class):
        """Normalized metrics should produce values ≤ 1 for reasonably distributed data."""
        logits, labels = random_5class
        for MetricCls in [
            ClassificationErrorRate,
            ClassificationCrossEntropy,
            ClassificationBrierScore,
        ]:
            m = MetricCls(normalize=True)
            m.update(logits, labels)
            val = m.compute().item()
            assert np.isfinite(val), f"{MetricCls.__name__} non-finite"

    def test_reset_clears_state(self, random_5class):
        logits, labels = random_5class
        m = ClassificationErrorRate(normalize=False)
        m.update(logits, labels)
        _ = m.compute()
        m.reset()
        with pytest.raises(ValueError, match="No data"):
            m.compute()
