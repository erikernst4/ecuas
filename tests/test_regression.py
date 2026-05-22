"""Regression tests for ECUAS metrics."""
import pytest
import torch
import numpy as np

from ecuas.classification import (
    ClassificationErrorRate, ClassificationCrossEntropy, ClassificationBrierScore,
    ClassificationAUC, ClassificationECE, ClassificationECUAS, ClassificationLogLog,
    ClassificationGammaECUAS, ClassificationAURC, ClassificationFPR95
)
from ecuas.confidences import (
    ConfidenceErrorRate, ConfidenceCrossEntropy, ConfidenceBrierScore,
    ConfidenceAUCScore, ExpectedCalibrationError, ConfidenceECUAS, CCAS,
    ConfidenceGammaECUAS, ConfidenceAURC, FPR95
)

def test_classification_regression():
    torch.manual_seed(42)
    logits = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    
    expected_values = {
        'ClassificationErrorRate': 0.75,
        'ClassificationErrorRate_norm': 1.1538461446762085,
        'ClassificationCrossEntropy': 1.410941243171692,
        'ClassificationCrossEntropy_norm': 1.287276268005371,
        'ClassificationBrierScore': 0.2806038558483124,
        'ClassificationBrierScore_norm': 1.2658820152282715,
        'ClassificationAUC': 0.5733333230018616,
        'ClassificationECE': 0.34274914860725403,
        'ClassificationECUAS': 1.2432149648666382,
        'ClassificationECUAS_norm': 1.2436069250106812,
        'ClassificationLogLog': 1.3122161626815796,
        'ClassificationLogLog_norm': 1.1944338083267212,
        'ClassificationGammaECUAS': 0.574999988079071,
        'ClassificationGammaECUAS_norm': 0.884615421295166,
        'ClassificationAURC': 0.7069567441940308,
        'ClassificationFPR95': 0.8666666746139526,
    }
    
    metrics = [
        ("ClassificationErrorRate", ClassificationErrorRate(normalize=False)),
        ("ClassificationErrorRate_norm", ClassificationErrorRate(normalize=True)),
        ("ClassificationCrossEntropy", ClassificationCrossEntropy(normalize=False)),
        ("ClassificationCrossEntropy_norm", ClassificationCrossEntropy(normalize=True)),
        ("ClassificationBrierScore", ClassificationBrierScore(normalize=False)),
        ("ClassificationBrierScore_norm", ClassificationBrierScore(normalize=True)),
        ("ClassificationAUC", ClassificationAUC()),
        ("ClassificationECE", ClassificationECE()),
        ("ClassificationECUAS", ClassificationECUAS(n=0, normalize=False)),
        ("ClassificationECUAS_norm", ClassificationECUAS(n=0, normalize=True)),
        ("ClassificationLogLog", ClassificationLogLog(normalize=False)),
        ("ClassificationLogLog_norm", ClassificationLogLog(normalize=True)),
        ("ClassificationGammaECUAS", ClassificationGammaECUAS(gamma=0.5, normalize=False)),
        ("ClassificationGammaECUAS_norm", ClassificationGammaECUAS(gamma=0.5, normalize=True)),
        ("ClassificationAURC", ClassificationAURC()),
        ("ClassificationFPR95", ClassificationFPR95()),
    ]
    
    for name, metric in metrics:
        metric.update(logits, labels)
        val = metric.compute().item()
        assert val == pytest.approx(expected_values[name], abs=1e-5), f"{name} changed from {expected_values[name]} to {val}"

def test_confidence_regression():
    torch.manual_seed(42)
    confidences = torch.rand(20)
    correctness = torch.randint(0, 2, (20,))
    
    expected_values = {
        'ConfidenceErrorRate': 0.3499999940395355,
        'ConfidenceCrossEntropy': 0.8088444471359253,
        'ConfidenceCrossEntropy_norm': 1.2492839097976685,
        'ConfidenceBrierScore': 0.28014951944351196,
        'ConfidenceBrierScore_norm': 2.462852954864502,
        'ConfidenceAUCScore': 0.5164835453033447,
        'ExpectedCalibrationError': 0.32705628871917725,
        'ConfidenceECUAS': 0.8030088543891907,
        'CCAS': 0.8030088543891907,
        'ConfidenceGammaECUAS': 0.44999998807907104,
        'ConfidenceAURC': 0.3707531690597534,
        'FPR95': 1.0,
    }
    
    metrics = [
        ("ConfidenceErrorRate", ConfidenceErrorRate()),
        ("ConfidenceCrossEntropy", ConfidenceCrossEntropy(normalize=False)),
        ("ConfidenceCrossEntropy_norm", ConfidenceCrossEntropy(normalize=True)),
        ("ConfidenceBrierScore", ConfidenceBrierScore(normalize=False)),
        ("ConfidenceBrierScore_norm", ConfidenceBrierScore(normalize=True)),
        ("ConfidenceAUCScore", ConfidenceAUCScore()),
        ("ExpectedCalibrationError", ExpectedCalibrationError()),
        ("ConfidenceECUAS", ConfidenceECUAS(n=0)),
        ("CCAS", CCAS()),
        ("ConfidenceGammaECUAS", ConfidenceGammaECUAS(gamma=0.5)),
        ("ConfidenceAURC", ConfidenceAURC()),
        ("FPR95", FPR95()),
    ]
    
    for name, metric in metrics:
        metric.update(confidences, correctness)
        val = metric.compute().item()
        assert val == pytest.approx(expected_values[name], abs=1e-5), f"{name} changed from {expected_values[name]} to {val}"
