"""NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss for SimCLR."""
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Vectorized NT-Xent loss function as defined in SimCLR (Chen et al. ICML 2020 Eq. 1).

    For a batch of N samples, two views produce 2N projections.
    Each sample i has 1 positive (its corresponding augmented view) and 2N-2 negatives.
    Cosine similarities are computed after L2 normalization and scaled by temperature tau.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: Optional[torch.Tensor] = None,
        return_acc: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """Compute the NT-Xent loss across all 2N representations.

        Args:
            z_i: Tensor of shape (N, D) or (2N, D).
            z_j: Optional Tensor of shape (N, D) if z_i is (N, D).
            return_acc: Whether to also return contrastive top-1 accuracy.

        Returns:
            loss: Scalar tensor containing mean contrastive cross-entropy loss over 2N anchors.
            acc (optional): Contrastive top-1 accuracy in [0.0, 1.0].
        """
        if z_j is not None:
            N = z_i.shape[0]
            z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        else:
            z = z_i
            total_samples = z.shape[0]
            if total_samples % 2 != 0:
                raise ValueError("Expected an even number of samples when passing single tensor")
            N = total_samples // 2

        # L2-normalize projection embeddings
        z = F.normalize(z, dim=1)

        # Pairwise cosine similarity matrix scaled by temperature: shape (2N, 2N)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create diagonal mask to exclude self-similarity
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

        # Ground truth positive targets:
        # Indices 0..N-1 pair with N..2N-1
        # Indices N..2N-1 pair with 0..N-1
        targets = torch.cat(
            [
                torch.arange(N, 2 * N, device=z.device),
                torch.arange(0, N, device=z.device),
            ],
            dim=0,
        )

        # Standard cross-entropy over the 2N-1 candidate negatives+positive per row
        loss = F.cross_entropy(sim_matrix, targets, reduction="mean")

        if return_acc:
            with torch.no_grad():
                preds = torch.argmax(sim_matrix, dim=1)
                acc = (preds == targets).float().mean().item()
            return loss, acc

        return loss


def nt_xent_loss_explicit_loop(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Slow, unvectorized reference loop implementation of NT-Xent loss.

    Directly computes Eq. 1 and Algorithm 1 for numerical verification in unit tests.
    """
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # 2N views: 0..N-1 are z_i, N..2N-1 are z_j
    z = F.normalize(z, dim=1)
    two_n = 2 * N

    total_loss = 0.0

    for i in range(two_n):
        # Determine positive index for anchor i
        pos_idx = (i + N) if i < N else (i - N)

        sim_i_pos = torch.dot(z[i], z[pos_idx]) / temperature
        num = torch.exp(sim_i_pos)

        denom = 0.0
        for k in range(two_n):
            if k != i:
                sim_i_k = torch.dot(z[i], z[k]) / temperature
                denom = denom + torch.exp(sim_i_k)

        term = -torch.log(num / denom)
        total_loss = total_loss + term

    return total_loss / two_n
