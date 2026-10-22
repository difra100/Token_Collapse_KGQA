"""The collapse metrics, defined once so stage 0, the training dynamics and the probe agree.

Formulas are token_probe.py's verbatim, so numbers from any of the three are comparable.

  stats(T, word_embedding_norm=None)
                        cos_inter, cv, eff_rank, resid_over_mean, norm_mean, n;
                        when supplied, also reports the direct absolute residual
                        magnitude in units of a typical word embedding.
  top1_retrieval(Z, T)  fraction whose own target is the argmax over all targets.
                        Needs Z and T in the same space (stage 0 aligns 1024-d encoder
                        output to 1024-d sentence embeddings, so this is exact there).
  nn_agreement(A, B)    dimension-free: fraction of samples whose nearest OTHER sample is
                        the same under A and under B. Used for the 4096-d graph token,
                        which cannot be compared to a 1024-d target directly; it answers
                        "does the token still order graphs the way the targets do".
"""
import torch
import torch.nn.functional as F


def stats(T: torch.Tensor, word_embedding_norm: torch.Tensor | float | None = None) -> dict:
    """Return token geometry statistics.

    ``resid_over_mean`` is scale-relative: it compares the mean residual to the
    common direction ``mu``. If ``word_embedding_norm`` is supplied (normally
    ``model.get_input_embeddings().weight.norm(dim=-1).median()``), the result
    additionally contains ``residual_over_word_embedding``:

        mean_i ||t_i - mu|| / word_embedding_norm.

    This is computed directly, rather than approximated as
    ``resid_over_mean * norm_mean / word_embedding_norm``.
    """
    T = T.float()
    n = T.size(0)
    Tn = F.normalize(T, dim=-1)
    mu = T.mean(0)
    cos = (Tn @ Tn.T).sum().sub(n).div(max(n * (n - 1), 1)).item()
    cv = (T.std(0).mean() / T.abs().mean()).item()
    s = torch.linalg.svdvals(T - mu)
    p = s / s.sum().clamp_min(1e-12)
    er = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    residual_norm_mean = (T - mu).norm(dim=-1).mean()
    out = dict(cos_inter=cos, cv=cv, eff_rank=er,
               resid_over_mean=(residual_norm_mean / mu.norm().clamp_min(1e-12)).item(),
               residual_norm_mean=residual_norm_mean.item(),
               norm_mean=T.norm(dim=-1).mean().item(), n=n)
    if word_embedding_norm is not None:
        ref = torch.as_tensor(word_embedding_norm, dtype=T.dtype, device=T.device)
        out["residual_over_word_embedding"] = (residual_norm_mean / ref.clamp_min(1e-12)).item()
    return out


def top1_retrieval(Z: torch.Tensor, T: torch.Tensor) -> float:
    if Z.size(0) != T.size(0) or Z.size(-1) != T.size(-1):
        raise ValueError(f"top1_retrieval needs matching shapes, got {tuple(Z.shape)} vs {tuple(T.shape)}")
    logits = F.normalize(Z.float(), dim=-1) @ F.normalize(T.float(), dim=-1).T
    return (logits.argmax(1) == torch.arange(Z.size(0), device=logits.device)).float().mean().item()


def nn_agreement(A: torch.Tensor, B: torch.Tensor) -> float:
    n = A.size(0)
    if n < 3:
        return float("nan")
    def nn(X):
        S = F.normalize(X.float(), dim=-1) @ F.normalize(X.float(), dim=-1).T
        S.fill_diagonal_(-2.0)              # exclude self
        return S.argmax(1)
    return (nn(A) == nn(B)).float().mean().item()
