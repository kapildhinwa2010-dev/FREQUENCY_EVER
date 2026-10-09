"""Frequency-domain ternary weight encoder and FFT-based linear layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class TernaryQuantizeSTE(torch.autograd.Function):
    """Straight-through estimator for {-1, 0, +1} quantization."""

    @staticmethod
    def forward(ctx, w):
        return torch.clamp(torch.round(w), -1, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


class FreqSignatureEncoder(nn.Module):
    """Encode ternary weights as wave signatures in frequency bins."""

    def __init__(self, num_bins: int = 8, neg_bin: int = 1, pos_bin: int = 2):
        super().__init__()
        self.num_bins = int(num_bins)
        self.neg_bin = int(neg_bin)
        self.pos_bin = int(pos_bin)

    def forward(self, w_ternary):
        sig = torch.zeros(*w_ternary.shape, self.num_bins, device=w_ternary.device, dtype=torch.cfloat)
        sig[..., self.neg_bin] = (w_ternary == -1).to(torch.cfloat)
        sig[..., self.pos_bin] = (w_ternary == +1).to(torch.cfloat)
        return sig

    def decode(self, sig):
        return sig[..., self.pos_bin].real - sig[..., self.neg_bin].real


class FreqBitNetLinear(nn.Module):
    """Linear layer using ternary weights encoded in a frequency signature."""

    def __init__(self, in_features, out_features, num_bins=8):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.w_real = nn.Parameter(torch.randn(out_features, in_features) * 0.8)
        self.enc = FreqSignatureEncoder(num_bins)

    def freq_weights(self):
        w_t = TernaryQuantizeSTE.apply(self.w_real)
        sig = self.enc(w_t)
        return sig, w_t

    def forward(self, x):
        sig, w_t = self.freq_weights()
        W_hard = self.enc.decode(sig)
        W = W_hard + (w_t - w_t.detach())
        return x @ W.T, W_hard

    def spectrum_energy(self):
        sig, _ = self.freq_weights()
        return sig.abs().pow(2).sum(dim=(0, 1))


__all__ = ["TernaryQuantizeSTE", "FreqSignatureEncoder", "FreqBitNetLinear"]
