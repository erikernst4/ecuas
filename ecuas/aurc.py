import torch
from torch_uncertainty.metrics.classification import AURC as _AURC
from torchmetrics.utilities.data import dim_zero_cat


class TieAveragedAURC(_AURC):
    """torch-uncertainty's AURC with the risk averaged within confidence ties.

    Samples with equal confidence cannot be separated by any threshold, so
    within a tie block the cumulative error count grows linearly with the
    block's mean error (the expectation over all orderings of the block).
    This makes the metric independent of the input order.
    """

    def partial_compute(self) -> torch.Tensor:
        """Selective risk at coverage k/N for k = 1..N, averaged within ties."""
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors).float()
        n = scores.numel()
        if n == 0:
            return errors
        order = scores.argsort(descending=True, stable=True)
        scores, errors = scores[order], errors[order]

        new_block = torch.ones(n, dtype=torch.bool, device=scores.device)
        new_block[1:] = scores[1:] != scores[:-1]
        block = new_block.cumsum(0) - 1
        n_blocks = int(block[-1]) + 1

        sizes = torch.zeros(n_blocks, dtype=errors.dtype, device=errors.device)
        sizes.index_add_(0, block, torch.ones_like(errors))
        block_errors = torch.zeros_like(sizes).index_add_(0, block, errors)
        errors_before = block_errors.cumsum(0) - block_errors
        starts = sizes.cumsum(0) - sizes

        k = torch.arange(1, n + 1, dtype=errors.dtype, device=errors.device)
        position_in_block = k - starts[block]
        cumulative_errors = errors_before[block] + block_errors[block] / sizes[block] * position_in_block
        return cumulative_errors / k
