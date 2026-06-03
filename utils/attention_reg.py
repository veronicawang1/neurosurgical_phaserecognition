"""
OperA-style attention regularization (MICCAI 2021).

L_reg = <n, CEE>  where n is normalized frame-wise attention (layer 0)
and CEE is per-frame cross-entropy of frozen CNN predictions vs labels.
"""

import torch
import torch.nn.functional as F


def load_feature_bundle(path: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Load a feature file. Supports:
      - dict with keys 'features' and optional 'cnn_logits'
      - legacy tensor [T, D]
    """
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        return data["features"], data.get("cnn_logits")
    return data, None


def normalized_frame_attention(
    attn_weights: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    OperA Eq. for n_j: sum_i A_ij / sum_i M_ij with causal mask M.

    attn_weights: [B, H, T, T] from first transformer layer
    padding_mask: [B, T], True where padded
    returns n: [B, T]
    """
    A = attn_weights.mean(dim=1)  # average heads -> [B, T, T]
    B, T, _ = A.shape

    M = torch.tril(torch.ones(T, T, device=A.device, dtype=A.dtype))
    col_sum = (A * M.unsqueeze(0)).sum(dim=1)
    denom = M.sum(dim=0).clamp(min=1e-8)
    n = col_sum / denom.unsqueeze(0)

    if padding_mask is not None:
        n = n.masked_fill(padding_mask, 0.0)

    return n


def compute_cee(
    cnn_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Per-frame cross-entropy (CEE) from frozen CNN. No gradient into CNN.

    cnn_logits: [B, T, C]
    labels: [B, T]
    returns cee: [B, T]
    """
    B, T, C = cnn_logits.shape
    cee = F.cross_entropy(
        cnn_logits.reshape(B * T, C),
        labels.reshape(B * T),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, T)
    return cee.detach()


def attention_regularization_loss(
    n: torch.Tensor,
    cee: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """L_reg = sum_j n_j * CEE_j over valid (non-padded) frames."""
    valid = labels != -100
    if valid.sum() == 0:
        return torch.tensor(0.0, device=n.device, dtype=n.dtype)
    return (n * cee * valid).sum() / valid.sum().float()


@torch.no_grad()
def cnn_logits_from_features(
    features: torch.Tensor,
    classifier: torch.nn.Linear,
) -> torch.Tensor:
    """Run frozen linear CNN head on pre-extracted features."""
    return classifier(features)
