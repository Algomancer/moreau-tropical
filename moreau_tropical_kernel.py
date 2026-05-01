
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _moreau_forward_kernel(
    x_ptr, W_ptr, lam,                  # x:(B,D), W:(N,D), λ scalar (fp32)
    y_ptr, tau_ptr, delta_ptr,           # y:(B,N), tau, delta: (B,N) fp32
    D,
    stride_xb, stride_xd,
    stride_wn, stride_wd,
    stride_yb, stride_yn,
    stride_tb, stride_tn,
    stride_db, stride_dn,
    BLOCK_D: tl.constexpr,
    N_BISECT: tl.constexpr,
):
    # one program per (b, n) — 2D grid avoids a runtime int-divmod by N
    b = tl.program_id(0)
    n = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load row x[b,:] and row W[n,:], promote to fp32 for stable bisection.
    x = tl.load(
        x_ptr + b * stride_xb + offs_d * stride_xd, mask=mask_d, other=0.0
    ).to(tl.float32)
    w = tl.load(
        W_ptr + n * stride_wn + offs_d * stride_wd, mask=mask_d, other=0.0
    ).to(tl.float32)
    z = w + x  # (BLOCK_D,) — register-resident

    z_safe_min = tl.where(mask_d, z, float("inf"))
    z_safe_max = tl.where(mask_d, z, float("-inf"))
    z_min = tl.min(z_safe_min, axis=0)
    z_max = tl.max(z_safe_max, axis=0)
    tau_lo = z_min - lam      # f(tau_lo) ≥ λ
    tau_hi = z_max            # f(tau_hi) = 0 ≤ λ

    for _ in tl.static_range(N_BISECT):
        tau_mid = 0.5 * (tau_lo + tau_hi)
        slack = tl.where(mask_d, tl.maximum(z - tau_mid, 0.0), 0.0)
        f = tl.sum(slack, axis=0)
        too_low = f > lam     # τ too low ⇒ slack too big ⇒ raise lower bound
        tau_lo = tl.where(too_low, tau_mid, tau_lo)
        tau_hi = tl.where(too_low, tau_hi, tau_mid)

    tau = 0.5 * (tau_lo + tau_hi)
    # Final slack & slack² (one extra reduce — cheap; alt is to fuse into the
    # last bisection iter at the cost of a branch).
    slack = tl.where(mask_d, tl.maximum(z - tau, 0.0), 0.0)
    slack_sq_sum = tl.sum(slack * slack, axis=0)
    # δ = (1/2λ) ∑ (z - τ)_+²    — saved separately so backward grad_λ avoids
    # the fp32 cancellation in (y - τ): δ has full fp32 *relative* precision,
    # whereas (y - τ) computed as a subtraction loses ~log10(|y|/|y-τ|) digits.
    delta = slack_sq_sum / (2.0 * lam)
    y = tau + delta

    tl.store(
        y_ptr + b * stride_yb + n * stride_yn,
        y.to(y_ptr.dtype.element_ty),
    )
    tl.store(tau_ptr + b * stride_tb + n * stride_tn, tau)
    tl.store(delta_ptr + b * stride_db + n * stride_dn, delta)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


_MAX_D = 2048
_DEFAULT_BISECT = 32


def moreau_tropical_forward(
    x: Tensor,
    W: Tensor,
    lam: Tensor | float,
    n_bisect: int = _DEFAULT_BISECT,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton-fused forward of the Moreau-tropical layer.

    Args:
        x: ``(B, D)`` cuda tensor (fp32; bf16/fp16 supported via cast inside).
        W: ``(N, D)`` cuda tensor, same dtype as ``x``.
        lam: scalar (positive) — Tensor or float.
        n_bisect: bisection iterations; the default ``32`` sits at the
            fp32-ulp agreement floor

    Returns:
        ``(y, tau, delta)`` where
            ``y``:     ``(B, N)`` in input dtype  — the layer output,
            ``tau``:   ``(B, N)`` fp32          — bisection threshold (for backward),
            ``delta``: ``(B, N)`` fp32          — ``(1/2λ) ∑ (z - τ)_+²`` saved for
                       ``grad_λ``; equals ``y - τ`` mathematically but stored
                       directly to avoid fp32 cancellation at small λ.

    Raises:
        ValueError on shape, device, dtype, or ``D > 2048`` violations.
    """
    if x.dim() != 2 or W.dim() != 2:
        raise ValueError(f"x and W must be 2D, got shapes {tuple(x.shape)} and {tuple(W.shape)}")
    if x.shape[1] != W.shape[1]:
        raise ValueError(f"feature mismatch: x.shape[1]={x.shape[1]} vs W.shape[1]={W.shape[1]}")
    if not (x.is_cuda and W.is_cuda):
        raise ValueError("x and W must both be on CUDA")
    if x.device != W.device:
        raise ValueError(f"x and W on different CUDA devices: {x.device} vs {W.device}")
    if x.dtype != W.dtype:
        raise ValueError(f"x and W must have the same dtype, got {x.dtype} vs {W.dtype}")

    B, D = x.shape
    N, _ = W.shape

    if D > _MAX_D:
        raise ValueError(f"D={D} > {_MAX_D} not supported in v0 (extend BLOCK_D tiling)")

    # λ → Python float for runtime kernel arg (no constexpr ⇒ no recompile per λ).
    lam_f = _as_pos_scalar(lam)

    BLOCK_D = triton.next_power_of_2(D)

    y = torch.empty((B, N), device=x.device, dtype=x.dtype)
    tau = torch.empty((B, N), device=x.device, dtype=torch.float32)
    delta = torch.empty((B, N), device=x.device, dtype=torch.float32)

    grid = (B, N)
    _moreau_forward_kernel[grid](
        x, W, lam_f,
        y, tau, delta,
        D,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        y.stride(0), y.stride(1),
        tau.stride(0), tau.stride(1),
        delta.stride(0), delta.stride(1),
        BLOCK_D=BLOCK_D,
        N_BISECT=n_bisect,
    )
    return y, tau, delta

@triton.jit
def _moreau_grad_x_kernel(
    grad_y_ptr,                          # (B, N)
    x_ptr, W_ptr, tau_ptr, lam,          # (B, D), (N, D), (B, N) fp32, scalar
    grad_x_ptr,                          # (B, D)
    N,
    stride_gyb, stride_gyn,
    stride_xb, stride_xd,
    stride_wn, stride_wd,
    stride_tb, stride_tn,
    stride_gxb, stride_gxd,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)

    xbd = tl.load(x_ptr + b * stride_xb + d * stride_xd).to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        gy = tl.load(
            grad_y_ptr + b * stride_gyb + offs_n * stride_gyn,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + offs_n * stride_wn + d * stride_wd,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        tau = tl.load(
            tau_ptr + b * stride_tb + offs_n * stride_tn,
            mask=mask_n, other=0.0,
        )

        slack = tl.maximum(w + xbd - tau, 0.0)
        p = slack / lam                                              # (BLOCK_N,)
        contrib = tl.where(mask_n, gy * p, 0.0)
        acc += tl.sum(contrib, axis=0)

    tl.store(
        grad_x_ptr + b * stride_gxb + d * stride_gxd,
        acc.to(grad_x_ptr.dtype.element_ty),
    )


def moreau_tropical_grad_x(
    x: Tensor,
    W: Tensor,
    lam: Tensor | float,
    tau: Tensor,
    grad_y: Tensor,
    block_n: int = 128,
) -> Tensor:

    _check_bwd_inputs(x, W, lam, tau, grad_y)
    B, D = x.shape
    N, _ = W.shape
    lam_f = _as_pos_scalar(lam)

    grad_x = torch.empty((B, D), device=x.device, dtype=x.dtype)
    grid = (B, D)
    _moreau_grad_x_kernel[grid](
        grad_y, x, W, tau, lam_f,
        grad_x,
        N,
        grad_y.stride(0), grad_y.stride(1),
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        tau.stride(0), tau.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_N=block_n,
    )
    return grad_x



@triton.jit
def _moreau_grad_W_kernel(
    grad_y_ptr,                          # (B, N)
    x_ptr, W_ptr, tau_ptr, lam,
    grad_W_ptr,                          # (N, D)
    B,
    stride_gyb, stride_gyn,
    stride_xb, stride_xd,
    stride_wn, stride_wd,
    stride_tb, stride_tn,
    stride_gwn, stride_gwd,
    BLOCK_B: tl.constexpr,
):
    n = tl.program_id(0)
    d = tl.program_id(1)

    # W[n, d] is reused inside the b-loop — scalar in regs.
    wnd = tl.load(W_ptr + n * stride_wn + d * stride_wd).to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)
    for b_start in range(0, B, BLOCK_B):
        offs_b = b_start + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        gy = tl.load(
            grad_y_ptr + offs_b * stride_gyb + n * stride_gyn,
            mask=mask_b, other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + offs_b * stride_xb + d * stride_xd,
            mask=mask_b, other=0.0,
        ).to(tl.float32)
        tau = tl.load(
            tau_ptr + offs_b * stride_tb + n * stride_tn,
            mask=mask_b, other=0.0,
        )
        slack = tl.maximum(wnd + x - tau, 0.0)
        p = slack / lam                                              # (BLOCK_B,)
        contrib = tl.where(mask_b, gy * p, 0.0)
        acc += tl.sum(contrib, axis=0)

    tl.store(
        grad_W_ptr + n * stride_gwn + d * stride_gwd,
        acc.to(grad_W_ptr.dtype.element_ty),
    )


def moreau_tropical_grad_W(
    x: Tensor,
    W: Tensor,
    lam: Tensor | float,
    tau: Tensor,
    grad_y: Tensor,
    block_b: int = 128,
) -> Tensor:
    _check_bwd_inputs(x, W, lam, tau, grad_y)
    B, D = x.shape
    N, _ = W.shape
    lam_f = _as_pos_scalar(lam)

    grad_W = torch.empty((N, D), device=W.device, dtype=W.dtype)
    grid = (N, D)
    _moreau_grad_W_kernel[grid](
        grad_y, x, W, tau, lam_f,
        grad_W,
        B,
        grad_y.stride(0), grad_y.stride(1),
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        tau.stride(0), tau.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        BLOCK_B=block_b,
    )
    return grad_W


# ---------------------------------------------------------------------------
# ∂L/∂λ  — no kernel; envelope identity collapses it to one PyTorch op
# ---------------------------------------------------------------------------


def moreau_tropical_grad_lam(
    grad_y: Tensor, delta: Tensor, lam: Tensor | float,
) -> Tensor:
    """``∂L/∂λ = -(grad_y · δ).sum() / λ``  with ``δ = slack²/(2λ)``  saved by the forward.

    Derived from ``∂y/∂λ = -½‖p*‖²`` (envelope theorem) and the forward identity
    ``‖p*‖² = 2δ/λ``. Using ``δ`` directly — instead of computing ``y - τ`` as a
    subtraction in fp32 — is essential at small λ: ``y - τ ≈ λ`` is dwarfed by
    ``y ≈ 1``, so the subtraction loses ``log10(1/λ)`` decimal digits.

    Args:
        grad_y: ``(B, N)`` cotangent, input dtype.
        delta:  ``(B, N)`` fp32 — second return of ``moreau_tropical_forward``.
        lam: scalar.

    Returns:
        Scalar fp32 tensor.
    """
    gy_f = grad_y.to(torch.float32)
    inner = (gy_f * delta).sum()                                    # ()
    if isinstance(lam, Tensor):
        return -inner / lam.to(torch.float32)
    return -inner / float(lam)



def _check_bwd_inputs(
    x: Tensor, W: Tensor, lam, tau: Tensor, grad_y: Tensor,
) -> None:
    if x.dim() != 2 or W.dim() != 2:
        raise ValueError(f"x, W must be 2D, got {tuple(x.shape)}, {tuple(W.shape)}")
    B, D = x.shape
    N, Dw = W.shape
    if D != Dw:
        raise ValueError(f"feature mismatch: D={D} vs {Dw}")
    if tau.shape != (B, N):
        raise ValueError(f"tau must be {(B, N)}, got {tuple(tau.shape)}")
    if grad_y.shape != (B, N):
        raise ValueError(f"grad_y must be {(B, N)}, got {tuple(grad_y.shape)}")
    if not (x.is_cuda and W.is_cuda and tau.is_cuda and grad_y.is_cuda):
        raise ValueError("all tensors must be on CUDA")
    if not (x.device == W.device == tau.device == grad_y.device):
        raise ValueError("tensors on different CUDA devices")
    if x.dtype != W.dtype or grad_y.dtype != x.dtype:
        raise ValueError(
            f"dtype mismatch: x={x.dtype}, W={W.dtype}, grad_y={grad_y.dtype}"
        )
    if tau.dtype != torch.float32:
        raise ValueError(f"tau must be fp32 (saved by forward), got {tau.dtype}")


def _as_pos_scalar(lam) -> float:
    if isinstance(lam, Tensor):
        if lam.numel() != 1:
            raise ValueError(f"lam must be scalar, got shape {tuple(lam.shape)}")
        v = float(lam.item())
    else:
        v = float(lam)
    if not v > 0:
        raise ValueError(f"lam must be positive, got {v}")
    return v


# ---------------------------------------------------------------------------
# autograd.Function wrap — drop-in for the reference's MoreauTropicalFn
# ---------------------------------------------------------------------------


import math
import torch.nn as nn


class MoreauTropicalKernelFn(torch.autograd.Function):
    """Triton-fused Moreau-tropical layer with envelope-theorem backward.

    """

    @staticmethod
    def forward(ctx, x: Tensor, W: Tensor, lam: Tensor) -> Tensor:
        if lam.dim() != 0:
            raise ValueError(f"lam must be 0-dim scalar tensor, got {tuple(lam.shape)}")
        y, tau, delta = moreau_tropical_forward(x, W, lam)
        ctx.save_for_backward(x, W, lam, tau, delta)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        x, W, lam, tau, delta = ctx.saved_tensors
        # All three calls are inside no_grad context (autograd is off in backward).
        grad_x = moreau_tropical_grad_x(x, W, lam, tau, grad_y)
        grad_W = moreau_tropical_grad_W(x, W, lam, tau, grad_y)
        grad_lam = moreau_tropical_grad_lam(grad_y, delta, lam)
        # Match upstream lam dtype (helper returns fp32; user lam may be other).
        if grad_lam.dtype != lam.dtype:
            grad_lam = grad_lam.to(lam.dtype)
        return grad_x, grad_W, grad_lam


# ---------------------------------------------------------------------------
# nn.Module wrap — mirrors moreau_tropical.MoreauTropical exactly
# ---------------------------------------------------------------------------


class MoreauTropicalKernel(nn.Module):
    """
    Shape:
        Input:  ``(B, D)`` or ``(D,)``
        Output: ``(B, N)`` or ``(N,)``
    """

    def __init__(self, in_features: int, out_features: int, lam: float = 1.0) -> None:
        super().__init__()
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("lam", torch.tensor(float(lam)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True
        y = MoreauTropicalKernelFn.apply(x, self.W, self.lam)
        return y.squeeze(0) if squeeze_back else y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"lam={self.lam.item():.4g}"
        )
