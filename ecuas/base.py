import torch
from torchmetrics import Metric as TorchMetricsMetric
from typing import Callable, Any

class Metric(TorchMetricsMetric):
    """
    Base class for ECUAS metrics.
    
    Inherits from torchmetrics.Metric (aliased as TorchMetricsMetric) and provides
    common boilerplate such as a default `_reduce` method and a classmethod
    for creating stateless shortcut functions.
    """
    full_state_update = False

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduces a given tensor based on the `self.reduction` attribute.
        
        Args:
            tensor (torch.Tensor): The tensor to reduce.
            
        Returns:
            torch.Tensor: The reduced tensor.
            
        Raises:
            AttributeError: If `self.reduction` is not set.
            ValueError: If `self.reduction` is invalid.
        """
        if not hasattr(self, "reduction"):
            raise AttributeError("Metric does not have a 'reduction' attribute.")
        
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, **default_kwargs) -> Callable[..., float]:
        """
        Creates a stateless shortcut function that instantiates the metric,
        updates it, and returns the computed scalar value.
        
        Args:
            **default_kwargs: Default keyword arguments used when instantiating the metric.
            
        Returns:
            Callable: A function that behaves as a one-shot metric evaluator.
        """
        def shortcut_function(*args, **kwargs) -> float:
            merged_kwargs = {**default_kwargs, **kwargs}
            metric = cls(**merged_kwargs)
            metric.update(*args)
            return metric.compute().item()

        return shortcut_function

class ClassificationMetric(Metric):
    """
    Base class for Classification Metrics that accumulate all logits and labels.
    """
    def __init__(self, normalize: bool = True, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Validates and accumulates logits and labels.
        """
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

class ConfidenceMetric(Metric):
    """
    Base class for Confidence Metrics that accumulates all confidences and correctness labels.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("all_confidences", default=[], dist_reduce_fx="cat")
        self.add_state("all_correctness", default=[], dist_reduce_fx="cat")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(correctness).any():
            raise ValueError("NaN values found in correctness tensor.")
        confidences = torch.where(
            torch.isnan(confidences), torch.tensor(0.5), confidences
        )
        self.all_confidences.append(confidences.float())
        self.all_correctness.append(correctness.float())
