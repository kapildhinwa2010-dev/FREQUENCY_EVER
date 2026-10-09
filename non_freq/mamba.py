"""FFT utilities and stable linear blocks used by the project."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_matmul(x, W):
    """Matrix multiply implemented through FFT when possible, with safe PyTorch fallback."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if W.dim() == 1:
        W = W.unsqueeze(1)
    if x.shape[-1] != W.shape[0]:
        raise ValueError(f"x last dim ({x.shape[-1]}) must match W rows ({W.shape[0]})")

    try:
        n = x.shape[-1]
        pad = max(2 * n, 4)
        X = torch.fft.rfft(x, n=pad, dim=-1)
        K = torch.fft.rfft(W.flip(0), n=pad, dim=0)
        if x.dim() == 2 and W.dim() == 2:
            Y = torch.fft.irfft(X[:, None, :] * K[None, :, :], n=pad, dim=-1)
            return Y[..., n - 1, :]
        Y = torch.fft.irfft(X.unsqueeze(-2) * K.unsqueeze(0), n=pad, dim=-1)
        return Y[..., n - 1, :]
    except Exception:
        return x @ W


def fft_matmul_b(x, W):
    """Batched matrix multiplication with FFT based implementation, fallback to einsum."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be [B, T, n], got {tuple(x.shape)}")
    if W.dim() != 3:
        raise ValueError(f"Expected W to be [B, n, m], got {tuple(W.shape)}")
    if x.shape[-1] != W.shape[1]:
        raise ValueError(f"x last dim {x.shape[-1]} != W inner dim {W.shape[1]}")
    try:
        B, T, n = x.shape
        _, _, m = W.shape
        pad = max(2 * n, 4)
        X = torch.fft.rfft(x, n=pad, dim=-1).unsqueeze(2)
        K = torch.fft.rfft(W.flip(1), n=pad, dim=1).transpose(1, 2)
        Y = torch.fft.irfft(X * K.unsqueeze(1), n=pad, dim=-1)
        return Y[..., n - 1, :]
    except Exception:
        return torch.einsum("btn,bnm->btm", x, W)


def fft_causal_conv(u, w):
    """Depthwise causal convolution using FFT."""
    if u.dim() != 3:
        raise ValueError(f"u must be [B, I, T], got {tuple(u.shape)}")
    if w.dim() != 2:
        raise ValueError(f"w must be [I, k], got {tuple(w.shape)}")
    B, I, T = u.shape
    k = w.shape[1]
    pad = T + k
    U = torch.fft.rfft(u, n=pad, dim=-1)
    Wk = torch.fft.rfft(w, n=pad, dim=-1)
    y = torch.fft.irfft(U * Wk[None], n=pad, dim=-1)
    y = y[..., k - 1:T]
    return F.pad(y, (k - 1, 0))


class FFTLinear(nn.Module):
    """Linear layer implemented through FFT-based matmul."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.W = nn.Parameter(torch.randn(in_features, out_features) * in_features ** -0.5)
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        y = fft_matmul(x, self.W)
        return y + self.b if self.b is not None else y


class FFTMatMulFn(torch.autograd.Function):
    """Custom autograd for FFT matmul."""

    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(x, W)
        return fft_matmul(x, W)

    @staticmethod
    def backward(ctx, grad_output):
        x, W = ctx.saved_tensors
        gx = fft_matmul(grad_output, W.T)
        x2 = x.reshape(-1, x.shape[-1])
        gy2 = grad_output.reshape(-1, grad_output.shape[-1])
        gW = gy2.T @ x2
        gW = gW.reshape(W.shape)
        return gx, gW


class FFTLinearFG(nn.Module):
    """Frequency-aware linear layer using FFTMatMulFn."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * in_features ** -0.5)
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        y = FFTMatMulFn.apply(x, self.W)
        return y + self.b if self.b is not None else y


def soliton_grad_pass(model, link, distance=1.0):
    """Apply a soliton-style pass over gradients."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.copy_(link(p.grad, distance))


def grad_spectrum(model, topk=3):
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.numel() > 8:
            g = p.grad.reshape(p.grad.shape[0], -1).mean(0)
            S = torch.fft.rfft(g).abs()
            out.append((name, torch.topk(S, topk).indices.tolist()))
    return out


class Mamba2BlockFFT(nn.Module):
    """A compact Mamba-2 style block with FFT-based projections."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        inner = d_model * expand
        self.inner = inner
        self.in_proj = FFTLinear(d_model, 2 * inner)
        self.conv_w = nn.Parameter(torch.randn(inner, d_conv) * 0.1)
        self.bc_proj = FFTLinear(inner, 2 * d_state)
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.out_proj = FFTLinear(inner, d_model)

    def forward(self, u):
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = fft_causal_conv(x.transpose(1, 2), self.conv_w).transpose(1, 2)
        xc = F.silu(xc)
        Bp, Cp = self.bc_proj(xc).chunk(2, dim=-1)
        lam = F.softplus(self.log_lambda)
        D = torch.einsum("btn,bnm->btm", Cp, Bp.transpose(1, 2))
        T = xc.shape[1]
        idx = torch.arange(T, device=xc.device).float()
        dt = idx[:, None] - idx[None, :]
        mask = torch.where(dt >= 0, torch.exp(-lam * dt.clamp(min=0)), torch.zeros_like(dt))
        Y = (D * mask[None]) @ xc
        return self.out_proj(Y * F.silu(z))


__all__ = [
    "fft_matmul",
    "fft_matmul_b",
    "fft_causal_conv",
    "FFTLinear",
    "FFTLinearFG",
    "FFTMatMulFn",
    "soliton_grad_pass",
    "grad_spectrum",
    "Mamba2BlockFFT",
]
