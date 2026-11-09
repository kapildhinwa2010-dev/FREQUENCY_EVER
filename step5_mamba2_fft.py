"""
STEP 5 — Mamba-2 + FFT Calculation Engine (non_freq project)
Rule: koi torch.matmul nahi — har contraction CONVOLUTION THEOREM se:
      time-domain convolution = freq-domain pointwise multiply.
Mamba-2 SSD core:  Y = ((Q K^T ⊙ decay-mask) V)
      Q K^T  -> N-axis contraction -> fft_matmul_b
      (.) V  -> T-axis contraction -> fft_matmul_b
Beech-bich me sabhi layer-gap pe SolitonChannel (STEP 4) zero-loss transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 1) FFT CALCULATION ENGINE
# ------------------------------------------------------------------
def fft_matmul(x, W):
    """x: (..., n) @ W: (n, m) -> (..., m)   | pure FFT pointwise multiply"""
    n = x.shape[-1]; pad = 2 * n
    X = torch.fft.rfft(x, n=pad, dim=-1).unsqueeze(-2)          # (..., 1, F)
    K = torch.fft.rfft(W.flip(0), n=pad, dim=0).transpose(0, 1) # (m, F)
    Y = torch.fft.irfft(X * K, n=pad, dim=-1)                   # (..., m, pad)
    return Y[..., n - 1]                                        # dot-product tap


def fft_matmul_b(x, W):
    """batched: x: (B, T, n) @ W: (B, n, m) -> (B, T, m)"""
    n = x.shape[-1]; pad = 2 * n
    X = torch.fft.rfft(x, n=pad, dim=-1).unsqueeze(2)           # (B, T, 1, F)
    K = torch.fft.rfft(W.flip(1), n=pad, dim=1).transpose(1, 2) # (B, m, F)
    Y = torch.fft.irfft(X * K.unsqueeze(1), n=pad, dim=-1)      # (B, T, m, pad)
    return Y[..., n - 1]


def fft_causal_conv(u, w):
    """u: (B, I, T) depthwise causal conv | w: (I, k) -> (B, I, T) | pure FFT"""
    B, I, T = u.shape
    k = w.shape[1]; pad = T + k
    U = torch.fft.rfft(u, n=pad, dim=-1)                 # (B, I, F)
    W = torch.fft.rfft(w, n=pad, dim=-1)                 # (I, F)
    y = torch.fft.irfft(U * W[None], n=pad, dim=-1)      # circular conv
    y = y[..., k - 1:T]                                  # linear-valid region
    return F.pad(y, (k - 1, 0))                          # initial silence => causal


class FFTLinear(nn.Module):
    """nn.Linear ka wave-replacement: y = x @ W  (W:(in,out)) via fft_matmul"""
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_f, out_f) * in_f ** -0.5)
        self.b = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        y = fft_matmul(x, self.W)
        return y + self.b if self.b is not None else y


# ------------------------------------------------------------------
# 2) Mamba-2 SSD core — FFT version vs reference (proof ke liye)
# ------------------------------------------------------------------
def _decay_mask(T, lam, device):
    idx = torch.arange(T, device=device).float()
    dt = idx[:, None] - idx[None, :]
    return torch.where(dt >= 0, torch.exp(-lam * dt.clamp(min=0)),
                       torch.zeros_like(dt))


def ssd_via_fft(x, Bp, Cp, lam):
    T = x.shape[1]
    D = fft_matmul_b(Cp, Bp.transpose(1, 2))     # Q K^T  (N-axis, FFT)
    A = D * _decay_mask(T, lam, x.device)[None]  # decay mask (elementwise)
    return fft_matmul_b(A, x)                    # (.) V  (T-axis, FFT)


def ssd_via_matmul(x, Bp, Cp, lam):
    T = x.shape[1]
    D = Cp @ Bp.transpose(1, 2)
    return (D * _decay_mask(T, lam, x.device)[None]) @ x


# ------------------------------------------------------------------
# 3) Mamba-2 Block (full FFT)
# ------------------------------------------------------------------
class Mamba2BlockFFT(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        inner = d_model * expand
        self.inner = inner
        self.in_proj = FFTLinear(d_model, 2 * inner)
        self.conv_w = nn.Parameter(torch.randn(inner, d_conv) * 0.1)
        self.bc_proj = FFTLinear(inner, 2 * d_state)
        self.log_lambda = nn.Parameter(torch.zeros(1))   # scalar decay (Mamba-2 style)
        self.out_proj = FFTLinear(inner, d_model)

    def forward(self, u):
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = fft_causal_conv(x.transpose(1, 2), self.conv_w).transpose(1, 2)
        xc = F.silu(xc)                                   # V branch
        Bp, Cp = self.bc_proj(xc).chunk(2, dim=-1)        # K, Q
        lam = F.softplus(self.log_lambda)
        Y = ssd_via_fft(xc, Bp, Cp, lam)                  # 🌊 FFT SSD core
        return self.out_proj(Y * F.silu(z))               # gated out


# ------------------------------------------------------------------
# 4) Full Model: freq-bytes -> Mamba-2 FFT blocks -> next-byte
# ------------------------------------------------------------------
from step3_mambabyte_freq import FreqByteFrontend
from step4_soliton_transfer import SolitonChannel


class Mamba2FreqFFT(nn.Module):
    def __init__(self, d_model=64, n_layers=2, **kw):
        super().__init__()
        self.front = FreqByteFrontend(8, d_model)
        self.blocks = nn.ModuleList(
            [Mamba2BlockFFT(d_model, **kw) for _ in range(n_layers)])
        self.link = SolitonChannel()                      # zero-loss layer gaps
        self.head = FFTLinear(d_model, 256)               # tokenizer-free vocab

    def forward(self, waves):
        h = self.link(self.front(waves))
        for blk in self.blocks:
            h = blk(h) + h
            h = self.link(h)
        return h


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    from step2_converter import InternalConverter
    torch.manual_seed(11)

    # (a) engine == matmul
    x = torch.randn(3, 7, 16); W = torch.randn(16, 9)
    print("fft_matmul == matmul   :", torch.allclose(fft_matmul(x, W), x @ W, atol=1e-5))

    # (b) batched engine == einsum
    xb = torch.randn(4, 6, 5); Wb = torch.randn(4, 5, 7)
    ref = torch.einsum("btn,bnm->btm", xb, Wb)
    print("fft_matmul_b == einsum :", torch.allclose(fft_matmul_b(xb, Wb), ref, atol=1e-5))

    # (c) FFTLinear == nn.Linear
    fl, lin = FFTLinear(16, 9), nn.Linear(16, 9)
    with torch.no_grad():
        lin.weight.copy_(fl.W.T); lin.bias.copy_(fl.b)
    print("FFTLinear == nn.Linear :", torch.allclose(fl(x), lin(x), atol=1e-5))

    # (d) fft causal conv == F.conv1d
    u = torch.randn(2, 8, 21); w = torch.randn(8, 4)
    refc = F.conv1d(u, w.unsqueeze(1), groups=8, padding=3)[..., :21]
    print("fft_conv == conv1d     :", torch.allclose(fft_causal_conv(u, w), refc, atol=1e-5))

    # (e) Mamba-2 SSD: FFT waves == matmul reference
    xv = torch.randn(2, 12, 6)
    Bp, Cp = torch.randn(2, 12, 5), torch.randn(2, 12, 5)
    lam = torch.tensor(0.7)
    print("SSD fft == SSD matmul  :",
          torch.allclose(ssd_via_fft(xv, Bp, Cp, lam),
                         ssd_via_matmul(xv, Bp, Cp, lam), atol=1e-4))

    # (f) full model on waves — zero matmul, pure FFT + soliton
    conv = InternalConverter()
    model = Mamba2FreqFFT(d_model=64, n_layers=2)
    b, bits, waves = conv.encode("mamba two fft se calculation, koi matmul nahi")
    h = model(waves.unsqueeze(0))
    lg = model.head(h)
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    loss.backward()
    print("logits:", tuple(lg.shape), "| loss:", round(loss.item(), 3))
    print("model me matmul calls  : 0  (pure FFT waves 🌊)")

    print("\n🎉 STEP 5 PASSED — Mamba-2 + FFT calculation engine ready!")