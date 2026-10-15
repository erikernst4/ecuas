# ECUAS: Expected Cost for Uncertainty-Augmented Systems

`ecuas` is a Python library containing popular calibration and classification evaluation metrics, alongside the principled ECUAS metric family, for Uncertainty-Augmented (UA) systems.

This library implements the **ECUAS<sub>n</sub>** metric family described in the paper:\
*[ECUAS<sub>n</sub>: A family of metrics for principled evaluation of uncertainty-augmented systems](https://arxiv.org/abs/2605.20490)*.

## Background

In high-stakes automated decision-making, access to predictive uncertainty is essential for enabling users to accept or reject predictions based on application-specific cost trade-offs.

Traditional evaluation approaches assess these systems using separate metrics for candidate predictions (e.g., accuracy) and uncertainty scores (e.g., AUC, ECE, Brier Score), or by integrating over risk-coverage curves (e.g., AURC) that ignore probabilistically interpretable uncertainties.

**ECUAS** solves this by providing a unified, decision-theory-based proper scoring rule (PSR) to comprehensively evaluate the task of interest. The parameter **$n$** controls the trade-off between the cost of incorrect predictions and imperfect uncertainties:
- A small value (e.g., $n=0$) heavily penalizes systems that give high confidence to incorrect answers, suitable for settings where accepting an incorrect answer has severe consequences.
- A large value (e.g., $n \rightarrow \infty$) acts more like the 0-1 cost error rate, giving milder penalties to confident-but-incorrect predictions.

ECUAS<sub>n</sub> is built from a simpler cost with a fixed **rejection cost $\gamma$**, also available as **γ-ECUAS**: a candidate prediction is rejected, at cost $\gamma$, when its uncertainty $u = 1 - \text{confidence}$ is larger than $\gamma$; otherwise it is accepted and pays the 0-1 cost (0 if correct, 1 if incorrect). Use γ-ECUAS when the rejection cost of your application is known; ECUAS<sub>n</sub> integrates this cost over all $\gamma$ with weight $w_n(\gamma) \propto \gamma^{n-1}$, for when it is not.

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
- **Confidence FPR95**: `FPR95`
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
- **Classification FPR95**: `ClassificationFPR95`

### Inputs
- **Confidence metrics** take `(confidences, correctness)`: a 1D tensor of confidences in $[0, 1]$ and a 1D boolean (or 0/1) tensor saying whether each prediction was correct.
- **Classification metrics** take `(logits, labels)`: a 2D tensor of shape `[N, K]` with **logits, not probabilities** (softmax is applied internally), and a 1D tensor of `N` integer labels.

### Normalized variants
Several metrics accept a `normalize` flag, which divides the metric by the value obtained by a system that only knows the class priors (or, for confidence metrics, the overall accuracy). Values below 1 indicate the system is better than that no-information baseline.

**Defaults differ:** classification error rate, cross-entropy, Brier score, ECUAS, LogLog and γ-ECUAS are **normalized by default** (`normalize=True`); pass `normalize=False` for the raw value. Confidence metrics are not normalized by default.

## Usage

### Confidence metrics

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

### Classification metrics

```python
import torch
from ecuas import ClassificationECUAS, ClassificationGammaECUAS

logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3], [0.1, 0.2, 0.3]])
labels = torch.tensor([0, 1, 0])

ecuas_metric = ClassificationECUAS(n=0, normalize=True)
ecuas_metric.update(logits, labels)
print(f"Normalized ECUAS (n=0): {ecuas_metric.compute().item():.4f}")

# Raw (unnormalized) γ-ECUAS with a rejection cost of 0.3
gamma_metric = ClassificationGammaECUAS(gamma=0.3, normalize=False)
gamma_metric.update(logits, labels)
print(f"γ-ECUAS (γ=0.3): {gamma_metric.compute().item():.4f}")
```

### One-shot functions

Every metric can build a stateless function that instantiates the metric, updates it, and returns a float:

```python
import torch
from ecuas import ConfidenceECUAS

confidences = torch.tensor([0.9, 0.8, 0.4, 0.9])
correctness = torch.tensor([True, True, False, False])

ecuas_0 = ConfidenceECUAS.create_shortcut_function(n=0)
print(ecuas_0(confidences, correctness))
```

### Metrics by id

`get_metric_from_id` looks up a metric by a string id, which is convenient for configuration files and experiment scripts:

```python
import torch
from ecuas.utils import get_metric_from_id

logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3], [0.1, 0.2, 0.3]])
labels = torch.tensor([0, 1, 0])

metric = get_metric_from_id("cls_ece_nbins=5")
print(metric["full_name"], metric["function"](logits, labels))
```

The returned dictionary contains:
- `function`: a one-shot function (see above).
- `obj`: a metric instance, for `update`/`compute`.
- `full_name`, `display`: names for tables and plots.
- `higher_is_better`: whether larger values are better.

Available ids (`conf_*` take confidences and correctness, `cls_*` take logits and labels):

| Metric | Confidence | Classification |
| --- | --- | --- |
| Error rate | `conf_error_rate` | `cls_error_rate`, `cls_ner` (normalized) |
| AUC | `conf_auc` | `cls_auc` |
| Brier score | `conf_brier`, `conf_norm_brier` | `cls_brier`, `cls_nbs` (normalized) |
| Cross-entropy | `conf_cross_entropy`, `conf_norm_cross_entropy` | `cls_cross_entropy`, `cls_nce` (normalized) |
| LogLog | | `cls_loglog`, `cls_norm_loglog` |
| ECE | `conf_ece`, `conf_ece_nbins=<int>` | `cls_ece`, `cls_ece_nbins=<int>` |
| ECUAS<sub>n</sub> | `conf_n-ecuas_n=<int>` | `cls_n-ecuas_n=<int>`, `cls_norm_n-ecuas_n=<int>` |
| γ-ECUAS | `conf_gamma-ecuas_gamma=<float>` | `cls_gamma-ecuas_gamma=<float>`, `cls_norm_gamma-ecuas_gamma=<float>` |
| AURC | `conf_aurc` | `cls_aurc` |
| FPR95 | `conf_fpr95` | `cls_fpr95` |

ECE uses 10 bins by default.

## Running Tests

Execute the unit test suite:
```bash
uv run pytest
```

## Citation

If you use this library, please cite the paper:

```bibtex
@misc{estienne2026ecuas,
  title={{ECUAS}$_n$: A family of metrics for principled evaluation of uncertainty-augmented systems},
  author={Estienne, Lautaro and Ernst, Erik and Vera, Mat{\'i}as and Piantanida, Pablo and Ferrer, Luciana},
  year={2026},
  eprint={2605.20490},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2605.20490}
}
```

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
