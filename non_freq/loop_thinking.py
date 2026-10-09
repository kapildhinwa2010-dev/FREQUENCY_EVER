"""Soliton-based information transfer channel."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SolitonChannel(nn.Module):
    """A lightweight soliton-inspired transfer channel."""

    def __init__(self, samples_per_slot=8, T0=1.0, beta2=-1.0, distance=1.0, steps=4, dynamic_range=3.0):
        super().__init__()
        self.S = samples_per_slot
        self.T0 = T0
        self.beta2 = beta2
        self.P0 = 1.0
        self.gamma = abs(beta2) / (self.P0 * T0 * T0)
        self.distance = distance
        self.steps = steps
        self.c = dynamic_range

    @staticmethod
    def _sech(z):
        return 1.0 / torch.cosh(z)

    def _grid(self, T, device):
        N = self.S * T
        tau = torch.arange(N, device=device).float()
        centers = torch.arange(T, device=device).float() * self.S + self.S / 2
        return tau, centers, N

    def encode(self, x):
        orig = x.shape
        if x.dim() != 3:
            x = x.reshape(1, -1, x.shape[-1])
        B, T, D = x.shape
        tau, centers, N = self._grid(T, x.device)
        template = self._sech((tau.view(1, 1, -1) - centers.view(1, T, 1)) / self.T0)
        phi = math.pi * 0.95 * torch.tanh(x / self.c)
        A = torch.exp(1j * phi).permute(0, 2, 1).unsqueeze(-1)
        psi = (A * template.unsqueeze(0)).reshape(B, D, N) * math.sqrt(self.P0)
        return psi, orig

    def propagate(self, psi, distance=None, steps=None):
        distance = self.distance if distance is None else distance
        steps = self.steps if steps is None else steps
        if distance == 0:
            return psi

        dz = distance / max(steps, 1)
        N = psi.shape[-1]
        w = (torch.fft.fftfreq(N, d=1.0, device=psi.device) * 2 * math.pi).view(1, 1, -1)
        lin = torch.exp(-1j * 0.5 * self.beta2 * w * w * (dz / 2))

        for _ in range(max(steps, 1)):
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=-1) * lin, dim=-1)
            psi = psi * torch.exp(1j * self.gamma * (psi.abs() ** 2) * dz)
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=-1) * lin, dim=-1)
        return psi

    def decode(self, psi, T, orig):
        _, centers, _ = self._grid(T, psi.device)
        samp = psi[:, :, centers.long()]
        gphase = self.gamma * self.P0 * self.distance / 2
        samp = samp * torch.exp(-1j * gphase)
        phi = torch.clamp(torch.angle(samp), -math.pi * 0.99, math.pi * 0.99)
        x = self.c * torch.atanh(torch.clamp(phi / (math.pi * 0.95), -0.999, 0.999))
        return x.permute(0, 2, 1).reshape(orig)

    def physics(self, x, distance=None):
        psi, orig = self.encode(x)
        psi = self.propagate(psi, distance)
        T = x.shape[1] if x.dim() == 3 else x.shape[-2]
        return self.decode(psi, T, orig)

    def forward(self, x, distance=None):
        with torch.no_grad():
            y = self.physics(x, distance)
        return y + (x - x.detach())

    def audit(self, x, distance=None):
        with torch.no_grad():
            y = self.physics(x, distance)
            psi, _ = self.encode(x)
            psiL = self.propagate(psi, distance)
            T = x.shape[1] if x.dim() == 3 else x.shape[-2]
            _, centers, _ = self._grid(T, x.device)
            amp = psiL[:, :, centers.long()].abs()
            return {
                "zero_loss_err": (y - x).abs().max().item(),
                "soliton_amplitude_integrity": (amp - math.sqrt(self.P0)).abs().max().item(),
                "energy_ratio": (psiL.abs().pow(2).sum() / psi.abs().pow(2).sum()).item(),
            }


class UnitaryWaveLink(nn.Module):
    """Backward-compatible unitary wave link for simple transfer."""

    def __init__(self, carrier=8.0, beta2=-1.0):
        super().__init__()
        self.carrier = carrier
        self.beta2 = beta2

    def _t(self, T, device):
        return torch.arange(T, device=device).float().view(1, T, 1)

    def _k(self, T, device):
        return torch.fft.fftfreq(T, d=1.0, device=device).view(1, T, 1) * 2 * math.pi

    def encode(self, x):
        return x.to(torch.cfloat) * torch.exp(1j * self.carrier * self._t(x.shape[1], x.device))

    def transfer(self, psi, dist):
        return torch.fft.ifft(
            torch.fft.fft(psi, dim=1) * torch.exp(-1j * 0.5 * self.beta2 * self._k(psi.shape[1], psi.device) ** 2 * dist),
            dim=1,
        )

    def decode(self, psi):
        return (psi * torch.exp(-1j * self.carrier * self._t(psi.shape[1], psi.device))).real

    def forward(self, x, dist=1.0):
        orig = x.shape
        x3 = x if x.dim() == 3 else x.reshape(1, -1, x.shape[-1])
        out = self.decode(self.transfer(self.transfer(self.encode(x3), dist), -dist))
        return out.reshape(orig)


def soliton_proof(T=512, T0=8.0, distance=64.0, steps=32):
    """Proof that the soliton channel preserves shape better than linear dispersion."""
    t = torch.arange(T).float()
    psi0 = (1.0 / torch.cosh((t - T / 2) / T0)).view(1, 1, -1).to(torch.cfloat)
    sol_ch = SolitonChannel(T0=T0)
    lin_ch = SolitonChannel(T0=T0)
    lin_ch.gamma = 0.0
    sol = sol_ch.propagate(psi0, distance, steps)
    lin = lin_ch.propagate(psi0, distance, steps)
    corr = lambda a, b: F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item()
    return corr(sol.abs(), psi0.abs()), corr(lin.abs(), psi0.abs())


__all__ = ["SolitonChannel", "UnitaryWaveLink", "soliton_proof"]
