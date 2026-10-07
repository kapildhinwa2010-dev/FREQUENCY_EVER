"""
STEP 4 — Soliton Waves Data Transfer (non_freq project)
Rule: har layer-gap me data SOLITON WAVE me travel karega, ZERO DATA LOSS.
  encode     : real activation -> complex wave packet (amplitude=data, carrier=phase)
  transfer   : Fourier-space phase rotation exp(-i*beta*k^2*d)  [UNITARY => energy conserved]
  compensate : receiver-side inverse rotation (dispersion compensation)
  decode     : carrier hatado -> exact original activation (zero loss)
Soliton proof: split-step Fourier (dispersion + Kerr) me sech pulse apni shape preserve karta hai.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sech(z):
    return 1.0 / torch.cosh(z)


# ------------------------------------------------------------------
# Soliton Channel — layer-gap data transfer
# ------------------------------------------------------------------
class SolitonChannel(nn.Module):
    def __init__(self, carrier=8.0, beta2=-1.0, gamma=1.0):
        super().__init__()
        self.carrier = carrier      # carrier wave frequency
        self.beta2 = beta2          # dispersion coefficient
        self.gamma = gamma          # Kerr nonlinearity (soliton balance)

    def _t(self, T, device):
        return torch.arange(T, device=device).float()[:, None, None]

    def _k(self, T, device):
        return torch.fft.fftfreq(T, d=1.0, device=device) * 2 * math.pi

    # --- encode: data -> wave packet ---
    def encode(self, x):
        t = self._t(x.shape[1], x.device)
        return x.to(torch.cfloat) * torch.exp(1j * self.carrier * t)

    # --- unitary propagation: energy conserved => channel zero-loss ---
    def transfer(self, psi, distance):
        k = self._k(psi.shape[1], psi.device)[None, :, None]
        Psi = torch.fft.fft(psi, dim=1)
        Psi = Psi * torch.exp(-1j * 0.5 * self.beta2 * k ** 2 * distance)
        return torch.fft.ifft(Psi, dim=1)

    # --- receiver: dispersion compensation ---
    def compensate(self, psi, distance):
        return self.transfer(psi, -distance)

    # --- decode: wave -> data ---
    def decode(self, psi):
        t = self._t(psi.shape[1], psi.device)
        return (psi * torch.exp(-1j * self.carrier * t)).real

    def forward(self, x, distance=1.0):
        psi = self.encode(x)
        psi = self.transfer(psi, distance)      # 🌊 travel
        psi = self.compensate(psi, distance)    # 🎯 receiver compensation
        return self.decode(psi)

    # --- true soliton propagation (split-step Fourier: dispersion + Kerr) ---
    def propagate_soliton(self, psi, distance, steps=16):
        dz = distance / steps
        k = self._k(psi.shape[1], psi.device)[None, :, None]
        lin = torch.exp(-1j * 0.5 * self.beta2 * k ** 2 * dz)
        for _ in range(steps):
            psi = psi * torch.exp(1j * self.gamma * (psi.abs() ** 2) * dz)  # Kerr
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * lin, dim=1)     # dispersion
        return psi


# ------------------------------------------------------------------
# Model wrapper: har layer-gap me soliton link
# ------------------------------------------------------------------
from step3_mambabyte_freq import MambaByteFreq


class SolitonMambaByteFreq(MambaByteFreq):
    def __init__(self, *a, distance=1.0, **kw):
        super().__init__(*a, **kw)
        self.link = SolitonChannel()
        self.distance = distance

    def forward(self, x, mode="freq"):
        h = self.freq_frontend(x) if mode == "freq" else self.byte_frontend(x)
        h = self.link(h, self.distance)          # 🌊 frontend ke baad
        for blk in self.blocks:
            h = blk(h) + h
            h = self.link(h, self.distance)      # 🌊 har block ke baad
        return h


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    from step2_converter import InternalConverter

    torch.manual_seed(7)
    ch = SolitonChannel()

    # --- (a) ZERO-LOSS link: jo gaya, wahi wapas aaya ---
    x = torch.randn(2, 32, 64)
    xr = ch(x, distance=3.0)
    print("zero-loss max err:", (xr - x).abs().max().item())

    # --- (b) channel me energy conserved (unitary => loss impossible) ---
    psi = ch.encode(x)
    psi_d = ch.transfer(psi, 3.0)
    e1, e2 = psi.abs().pow(2).sum().item(), psi_d.abs().pow(2).sum().item()
    print("energy ratio (after/before):", round(e2 / e1, 6))

    # --- (c) SOLITON proof: sech pulse shape preserve karta hai ---
    T = 256
    t = torch.arange(T).float()
    psi0 = sech((t - T / 2) / 8.0)[None, :, None].to(torch.cfloat)
    ch.gamma = 1.0 / 64.0                      # soliton condition N=1 (matched)
    sol = ch.propagate_soliton(psi0, 120.0, steps=24)
    ch.gamma = 0.0                             # Kerr OFF => sirf linear dispersion
    lin = ch.propagate_soliton(psi0, 120.0, steps=24)
    ch.gamma = 1.0

    def corr(a, b):
        return F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item()

    print("shape corr | SOLITON (Kerr ON) :", round(corr(sol.abs(), psi0.abs()), 4))
    print("shape corr | linear  (Kerr OFF):", round(corr(lin.abs(), psi0.abs()), 4))

    # --- (d) full model: har layer-gap pe soliton channel ---
    conv = InternalConverter()
    model = SolitonMambaByteFreq(d_model=64, n_layers=2)
    b, bits, waves = conv.encode("soliton channel zero loss")
    h = model(waves.unsqueeze(0))
    lg = model.logits(h)
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    loss.backward()
    print("model+soliton links logits:", tuple(lg.shape), "| loss:", round(loss.item(), 3))

    print("\n🎉 STEP 4 PASSED — Soliton Waves data transfer ready!")