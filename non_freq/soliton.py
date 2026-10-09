"""Mamba-inspired sequence blocks and frequency frontends."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqByteFrontend(nn.Module):
    """Convert complex wave inputs to hidden vectors."""

    def __init__(self, num_carriers=8, d_model=64):
        super().__init__()
        self.num_carriers = int(num_carriers)
        self.proj = nn.Linear(2 * num_carriers, d_model)

    def forward(self, waves):
        if waves.dim() != 3:
            raise ValueError(f"waves must be [B, T, 8], got {tuple(waves.shape)}")
        z = torch.cat([waves.real, waves.imag], dim=-1)
        return self.proj(z)


class ByteFrontend(nn.Module):
    """Classic byte-id embedding frontend."""

    def __init__(self, d_model=64, vocab=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)

    def forward(self, ids):
        return self.emb(ids.long())


def selective_scan(x, dt, A, Bp, Cp, D):
    """Minimal selective scan implementation."""
    Bz, T, I = x.shape
    N = A.shape[1]
    h = torch.zeros(Bz, I, N, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(T):
        dA = torch.exp(-dt[:, t, :].unsqueeze(-1) * A)
        dB = dt[:, t, :].unsqueeze(-1) * Bp[:, t, :].unsqueeze(1)
        h = h * dA + dB * x[:, t, :].unsqueeze(-1)
        ys.append((h * Cp[:, t, :].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=1)
    return y + x * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """Simple Mamba block using selective scan."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_state = d_state
        inner = d_model * expand
        self.inner = inner
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, 2 * d_state + inner, bias=False)
        self.A = nn.Parameter(torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, u):
        T = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        xc = F.silu(xc)
        Bp, Cp, dt = self.x_proj(xc).split([self.d_state, self.d_state, self.inner], dim=-1)
        dt = F.softplus(dt)
        y = selective_scan(xc, dt, self.A, Bp, Cp, self.D)
        return self.out_proj(y * F.silu(z))


class MambaByteFreq(nn.Module):
    """Full model combining byte and wave frontends."""

    def __init__(self, d_model=64, n_layers=2, d_state=16, d_conv=4, expand=2, num_carriers=8, byte_vocab=256):
        super().__init__()
        self.freq_frontend = FreqByteFrontend(num_carriers, d_model)
        self.byte_frontend = ByteFrontend(d_model, byte_vocab)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, byte_vocab)

    def forward(self, x, mode="freq"):
        if mode == "freq":
            h = self.freq_frontend(x)
        elif mode == "byte":
            h = self.byte_frontend(x)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        for blk in self.blocks:
            h = blk(h) + h
        return h

    def logits(self, h):
        return self.head(h)


__all__ = ["FreqByteFrontend", "ByteFrontend", "MambaBlock", "MambaByteFreq"]
