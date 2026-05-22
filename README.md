# ECUAS: Expected Cost for Uncertainty-Augmented Systems

`ecuas` is a Python library containing popular calibration and classification evaluation metrics, alongside the principled ECUAS metric family, for Uncertainty-Augmented (UA) systems.

This library implements the **ECUAS_n** metric family described in the paper:
*[ECUAS_n: A family of metrics for principled evaluation of uncertainty-augmented systems](https://arxiv.org/abs/2605.20490v1)*.

## Background

In high-stakes automated decision-making, access to predictive uncertainty is essential for enabling users to accept or reject predictions based on application-specific cost trade-offs.

Traditional evaluation approaches assess these systems using separate metrics for candidate predictions (e.g., accuracy) and uncertainty scores (e.g., AUC, ECE, Brier Score), or by integrating over risk-coverage curves (e.g., AURC) that ignore probabilistically interpretable uncertainties.

**ECUAS** solves this by providing a unified, decision-theory-based proper scoring rule (PSR) to comprehensively evaluate the task of interest. The parameter **$n$** controls the trade-off between the cost of incorrect predictions and imperfect uncertainties:
- A small value (e.g., $n=0$) heavily penalizes systems that give high confidence to incorrect answers, suitable for settings where accepting an incorrect answer has severe consequences.
- A large value (e.g., $n \rightarrow \infty$) acts more like the 0-1 cost error rate, giving milder penalties to confident-but-incorrect predictions.

## Installation

### Via `uv` (Recommended)
Add `ecuas` directly to your project:
```bash
uv add ecuas
```

### Via `pip`
You can install `ecuas` from PyPI:
```bash
pip install ecuas
```

## Features and Metrics

### Confidence/Selective Prediction Metrics
- **Expected Calibration Error (ECE)**: `ExpectedCalibrationError`
- **Confidence Error Rate**: `ConfidenceErrorRate`
- **Confidence AUC Score**: `ConfidenceAUCScore`
- **Confidence Brier Score**: `ConfidenceBrierScore`
- **Confidence Cross-Entropy**: `ConfidenceCrossEntropy`
- **Confidence ECUAS** (n-ECUAS): `ConfidenceECUAS`
- **Confidence Gamma-ECUAS**: `ConfidenceGammaECUAS`
- **Confidence AURC**: `ConfidenceAURC`
- **CCAS** (Confidence Cost for Selective Prediction): `CCAS`

### Classification Metrics
- **Classification Error Rate**: `ClassificationErrorRate`
- **Classification Cross-Entropy**: `ClassificationCrossEntropy`
- **Classification Brier Score**: `ClassificationBrierScore`
- **Classification AUC**: `ClassificationAUC`
- **Classification ECE**: `ClassificationECE`
- **Classification ECUAS**: `ClassificationECUAS`
- **Classification LogLog**: `ClassificationLogLog`
- **Classification Gamma-ECUAS**: `ClassificationGammaECUAS`
- **Classification AURC**: `ClassificationAURC`

## Usage Example

```python
import torch
from ecuas import ConfidenceECUAS, ExpectedCalibrationError

# Setup data
confidences = torch.tensor([0.9, 0.8, 0.4, 0.9])
correctness = torch.tensor([True, True, False, False])

# Expected Calibration Error
ece_metric = ExpectedCalibrationError(n_bins=10)
ece_metric.update(confidences, correctness)
ece_val = ece_metric.compute()
print(f"ECE: {ece_val.item():.4f}")

# Confidence n-ECUAS (e.g., n=0 to heavily penalize overconfident errors)
ecuas_metric = ConfidenceECUAS(n=0)
ecuas_metric.update(confidences, correctness)
ecuas_val = ecuas_metric.compute()
print(f"ECUAS (n=0): {ecuas_val.item():.4f}")
```

## Running Tests

Execute the unit test suite:
```bash
uv run pytest
```

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
