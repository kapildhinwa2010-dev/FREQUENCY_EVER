"""Loop-thinking / ACT style module used by the unified model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .fft import FFTLinear, Mamba2BlockFFT
from .soliton import SolitonChannel


class HaltGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = FFTLinear(d, 1)

    def forward(self, h):
        return torch.sigmoid(self.g(h))


class LoopThinking(nn.Module):
    def __init__(self, d_model, max_loops=6, **kw):
        super().__init__()
        self.thought = Mamba2BlockFFT(d_model, **kw)
        self.gate = HaltGate(d_model)
        self.link = SolitonChannel()
        self.max_loops = max_loops

    def forward(self, h):
        B, T, D = h.shape
        running = torch.ones(B, T, 1, device=h.device)
        accum = torch.zeros_like(h)
        ponder = torch.zeros(B, T, 1, device=h.device)
        trace = []

        for t in range(self.max_loops):
            h = self.link(self.thought(h) + h)
            p = self.gate(h) * running
            if t == self.max_loops - 1:
                p = running
            accum = accum + p * h
            ponder = ponder + p * (t + 1)
            running = running - p
            trace.append(float(p.mean().item()))
        return accum, ponder, trace


class NonFreqThinker(nn.Module):
    def __init__(self, d_model=64, n_layers=2, max_loops=6, **kw):
        super().__init__()
        self.front = nn.Linear(8 * 2, d_model)
        self.blocks = nn.ModuleList([Mamba2BlockFFT(d_model, **kw) for _ in range(n_layers)])
        self.link = SolitonChannel()
        self.loop = LoopThinking(d_model, max_loops, **kw)
        self.head = FFTLinear(d_model, 256)

    def forward(self, waves, return_thoughts=False):
        if waves.dim() == 2:
            waves = waves.unsqueeze(0)
        h = self.link(self.front(torch.cat([waves.real, waves.imag], dim=-1)))
        for blk in self.blocks:
            h = self.link(blk(h) + h)
        h_th, ponder, trace = self.loop(h)
        logits = self.head(h_th)
        return (logits, ponder, trace) if return_thoughts else logits


__all__ = ["HaltGate", "LoopThinking", "NonFreqThinker"]
