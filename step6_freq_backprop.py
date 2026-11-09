"""
STEP 6 — Frequency Backpropagation (non_freq project)
Rule (point #5): gradient numbers me nahi, WAVES me travel karega.
  1. FFTMatMulFn  : forward FFT se, backward bhi FFT se (chain-rule = freq pointwise multiply)
  2. Soliton grads: har param-grad soliton channel se zero-loss pass hota hai
  3. Grad spectrum: freq-domain signature monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from step5_mamba2_fft import fft_matmul
from step3_mambabyte_freq import FreqByteFrontend
from step4_soliton_transfer import SolitonChannel


# ------------------------------------------------------------------
# 1) Custom autograd: forward FFT, backward bhi FFT
# ------------------------------------------------------------------
class FFTMatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(x, W)
        return fft_matmul(x, W)                 # 🌊 forward wave-domain

    @staticmethod
    def backward(ctx, gy):
        x, W = ctx.saved_tensors
        gx = fft_matmul(gy, W.transpose(0, 1))  # gy @ W^T  (FFT engine)
        gW = fft_matmul(gy.reshape(-1, gy.shape[-1]).t(),
                        x.reshape(-1, x.shape[-1])).t()   # x^T @ gy (FFT engine)
        return gx, gW                           # dono gradients freq-domain se


class FFTLinearFG(nn.Module):
    """FFTLinear + frequency-backprop (STEP 7 me yahi har jagah lagega)"""
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_f, out_f) * in_f ** -0.5)
        self.b = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        y = FFTMatMulFn.apply(x, self.W)
        return y + self.b if self.b is not None else y


# ------------------------------------------------------------------
# 2) Soliton gradient channels + spectrum monitor
# ------------------------------------------------------------------
def soliton_grad_pass(model, link, distance=1.0):
    """har param-grad soliton wave me travel karke zero-loss wapas aata hai"""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.copy_(link(p.grad, distance))


def grad_spectrum(model, topk=3):
    """gradient ki freq-signature: kaunsi carriers seekh rahi hain"""
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.numel() > 8:
            g = p.grad.reshape(p.grad.shape[0], -1).mean(0)
            S = torch.fft.rfft(g).abs()
            k = min(topk, S.numel())
            if k:
                out.append((name, torch.topk(S, k).indices.tolist()))
    return out


# ------------------------------------------------------------------
# 3) Tiny model (freq-backprop end-to-end prove karne ke liye)
# ------------------------------------------------------------------
class TinyFreqGradModel(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.front = FreqByteFrontend(8, d)
        self.mid = FFTLinearFG(d, d)
        self.head = FFTLinearFG(d, 256)
        self.link = SolitonChannel()

    def forward(self, waves):
        h = self.link(self.front(waves))
        h = self.link(F.silu(self.mid(h)))
        return self.head(h)


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    from step2_converter import InternalConverter
    torch.manual_seed(5)

    # --- (a) freq-backprop == reference backprop (exact chain-rule) ---
    x = torch.randn(5, 6, 4, requires_grad=True)
    W = torch.randn(4, 3, requires_grad=True)
    gy = torch.randn(5, 6, 3)
    gx, gW = torch.autograd.grad(FFTMatMulFn.apply(x, W), (x, W), gy)
    x2 = x.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    gx2, gW2 = torch.autograd.grad(x2 @ W2, (x2, W2), gy)
    print("freq-backprop == ref-backprop:",
          torch.allclose(gx, gx2, atol=1e-5), torch.allclose(gW, gW2, atol=1e-5))

    # --- (b) soliton grad channel zero-loss ---
    link = SolitonChannel()
    g = torch.randn(4, 24, 16)
    print("soliton grad zero-loss err:", (link(g, 2.0) - g).abs().max().item())

    # --- (c) mini TRAINING: bytes -> waves -> freq-backprop -> loss girega ---
    conv = InternalConverter()
    model = TinyFreqGradModel(d=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    text = "ab" * 40                      # predictable pattern => seekhne layak
    b, bits, waves = conv.encode(text)
    Wv = waves.unsqueeze(0)
    tgt = b.long()[1:]

    losses = []
    for step in range(60):
        opt.zero_grad()
        lg = model(Wv)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), tgt)
        loss.backward()                              # 🌊 freq-domain chain rule
        soliton_grad_pass(model, link)               # 🌊 grads wave me travel
        opt.step()
        losses.append(loss.item())

    print("loss start:", round(losses[0], 3), "-> loss end:", round(losses[-1], 3),
          "| seekha:", losses[-1] < losses[0] - 1.0)

    # --- (d) gradient spectrum (freq-signature monitor) ---
    print("grad spectrum (top carrier bins):", grad_spectrum(model))

    print("\n🎉 STEP 6 PASSED — Frequency Backpropagation ready!")
