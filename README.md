# ECUAS: Expected Cost for Uncertainty-Augmented Systems

`ecuas` is a Python library containing robust calibration and classification evaluation metrics for Uncertainty-Augmented systems. It implements expected cost metrics, ECE, Brier Score, Cross Entropy, AUC, CCAS, LogLog, and selective prediction metrics under a unified `torchmetrics.Metric` interface.

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

# Confidence n-ECUAS
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
