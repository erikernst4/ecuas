import torch
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
import torch.nn.functional as F
from torch_uncertainty.metrics.classification import AURC as _AURC, FPRx as _FPRx

from .base import ConfidenceMetric, Metric


class ExpectedCalibrationError(BinaryCalibrationError, Metric):
    """ECE via torchmetrics BinaryCalibrationError (L1 norm)."""

    def __init__(self, n_bins: int = 10, **kwargs):
        super().__init__(n_bins=n_bins, norm="l1", **kwargs)

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(correctness).any():
            raise ValueError("NaN values found in correctness tensor.")
        confidences = torch.where(
            torch.isnan(confidences), torch.tensor(0.5), confidences
        )
        super().update(confidences.float(), correctness.long())


class ConfidenceErrorRate(ConfidenceMetric):
    """
    Error Rate for confidence predictions.
    Error Rate = mean[ y != argmax(p) ]
    """

    def compute(self) -> torch.Tensor:
        if not self.all_confidences:
            raise ValueError("No samples to compute error rate.")
        correctness = torch.cat(self.all_correctness)
        return (correctness == 0).float().mean()


class ConfidenceAUCScore(BinaryAUROC, Metric):
    """AUROC via torchmetrics BinaryAUROC."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(correctness).any():
            raise ValueError("NaN values found in correctness tensor.")
        confidences = torch.where(
            torch.isnan(confidences), torch.tensor(0.5), confidences
        )
        super().update(confidences.float(), correctness.long())

    def compute(self) -> torch.Tensor:
        try:
            return super().compute()
        except (ValueError, IndexError):
            raise ValueError("No samples to compute AUC score.")


class ConfidenceBrierScore(MeanSquaredError, Metric):
    """Brier Score = MSE between confidence and correctness."""

    def __init__(self, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        if normalize:
            self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(correctness).any():
            raise ValueError("NaN values found in correctness tensor.")
        confidences = torch.where(
            torch.isnan(confidences), torch.tensor(0.5), confidences
        )
        super().update(confidences.float(), correctness.float())
        if self.normalize:
            self.sum_corr += correctness.float().sum()
            self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        mse = super().compute()
        if self.normalize:
            corr_prior = self.sum_corr / self.count
            mse_prior = corr_prior * (1 - corr_prior) / 2
            mse = mse / mse_prior
        return mse


class ConfidenceCrossEntropy(ConfidenceMetric):
    """
    Binary Cross Entropy between confidence and correctness.
    CE = -mean[ y·log(p) + (1-y)·log(1-p) ]
    """

    def __init__(self, epsilon: float = 1e-7, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.normalize = normalize

    def compute(self) -> torch.Tensor:
        if not self.all_confidences:
            raise ValueError("No samples to compute cross entropy.")
        conf = torch.cat(self.all_confidences).clamp(self.epsilon, 1 - self.epsilon)
        corr = torch.cat(self.all_correctness)
        ce = F.binary_cross_entropy(conf, corr, reduction="mean")
        if self.normalize:
            corr_prior = corr.mean()
            ce_prior = -(
                corr_prior * torch.log(corr_prior + self.epsilon)
                + (1 - corr_prior) * torch.log(1 - corr_prior + self.epsilon)
            )
            ce = ce / ce_prior
        return ce


class ConfidenceECUAS(ConfidenceMetric):
    """ECUAS (Expected Cost for Uncertainty-Augmented Systems) metric."""

    def __init__(self, n: int = 0, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.epsilon = epsilon
        if n == 0:
            self.cost_fun = lambda q, correct_indicator: torch.where(
                condition=correct_indicator.bool(),
                input=1 - q,
                other=1 - q - torch.log(1 - q),
            )
        elif n > 0:
            self.cost_fun = lambda q, correct_indicator: torch.where(
                condition=correct_indicator.bool(),
                input=(1 - q) ** (n + 1),
                other=(1 - q) ** (n + 1) + (n + 1) / n * (1 - (1 - q) ** n),
            )
        else:
            raise ValueError("n must be non-negative.")

    def compute(self) -> torch.Tensor:
        if not self.all_confidences:
            raise ValueError("No samples to compute ECUAS.")
        q = torch.cat(self.all_confidences).clamp(self.epsilon, 1 - self.epsilon)
        indicator = torch.cat(self.all_correctness)
        return self.cost_fun(q, indicator).mean()


class ConfidenceGammaECUAS(ConfidenceMetric):
    """Gamma-ECUAS (Expected Cost for Uncertainty-Augmented Systems) metric."""

    def __init__(self, gamma: float = 0.5, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.epsilon = epsilon

    def compute(self) -> torch.Tensor:
        if not self.all_confidences:
            raise ValueError("No samples to compute Gamma-ECUAS.")
        confidences = torch.cat(self.all_confidences)
        correctness = torch.cat(self.all_correctness)
        s = 1.0 - confidences
        abstain_mask = s < self.gamma
        answer_mask = ~abstain_mask
        total_cost = torch.zeros_like(confidences)
        total_cost[abstain_mask] = self.gamma
        total_cost[answer_mask] = 1.0 - correctness[answer_mask]
        return total_cost.mean()


class CCAS(ConfidenceECUAS):
    def __init__(self, *args, **kwargs):
        super().__init__(n=0, *args, **kwargs)


class ConfidenceAURC(_AURC, Metric):
    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        probs = torch.cat([1 - confidences.view(-1, 1), confidences.view(-1, 1)], dim=1)
        return super().update(probs, correctness.long())


class FPR95(_FPRx, Metric):
    def __init__(self, **kwargs):
        super().__init__(recall_level=0.95, pos_label=1, **kwargs)

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        return super().update(confidences, correctness.long())
