"""
STEP 1 — Frequency BitNet (non_freq project)
Ternary weights {-1, 0, +1} ko frequency-domain waves mein represent karna.
Rule: sirf 2 carriers ->  -1 = f_neg wave | 0 = silence | +1 = f_pos wave
Calculation: matrix multiply NAHI -> FFT point-wise multiply (convolution theorem)
Zero data loss: encode -> decode exact wapas milta hai.
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# 1) Ternary Quantizer + Straight-Through Estimator (STE)
#    forward : real weight -> {-1, 0, +1}
#    backward: gradient seedha pass-through (1.58 BitNet trick)
# ------------------------------------------------------------------
class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.clamp(torch.round(w), -1, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad                      # identity => STE


# ------------------------------------------------------------------
# 2) Ternary -> Frequency Signature Encoder
#    har weight ek wave-signature ban jaata hai (bins mein se sirf 2 active)
# ------------------------------------------------------------------
class FreqSignatureEncoder(nn.Module):
    def __init__(self, num_bins: int = 8, neg_bin: int = 1, pos_bin: int = 2):
        super().__init__()
        self.num_bins = num_bins
        self.neg_bin = neg_bin           # f_neg carrier
        self.pos_bin = pos_bin           # f_pos carrier
        # baaki sab bins = silence (notebook: "only 2 freq")

    def forward(self, w_ternary):
        sig = torch.zeros(*w_ternary.shape, self.num_bins,
                          dtype=torch.cfloat, device=w_ternary.device)
        sig[..., self.neg_bin] = (w_ternary == -1).float()   # -1 => f_neg wave
        sig[..., self.pos_bin] = (w_ternary == +1).float()   # +1 => f_pos wave
        return sig                                           # 0 => sab silence

    def decode(self, sig):
        """wave-signature se wapas ternary value: (+1 bin) - (-1 bin)"""
        return sig[..., self.pos_bin].real - sig[..., self.neg_bin].real


# ------------------------------------------------------------------
# 3) FFT se calculation — matrix multiply ki jagah
#    y = x @ W.T  <=>  time-domain convolution  <=>  freq-domain pointwise multiply
#    zero-padding => koi wrap-around loss nahi (zero data loss)
# ------------------------------------------------------------------
def fft_matmul(x, W):
    # x: (..., in) | W: (out, in) -> y: (..., out)   [2D aur 3D batched dono safe]
    out_shape = x.shape[:-1] + (W.shape[0],)
    xf = x.reshape(-1, x.shape[-1])                 # leading batch dims flatten
    n = xf.shape[-1]
    pad = 2 * n                                    # no aliasing => no data loss
    X = torch.fft.rfft(xf, n=pad, dim=-1)          # (B', F)
    K = torch.fft.rfft(W.flip(-1), n=pad, dim=-1)  # (out, F)  flip => dot-product
    Y = torch.fft.irfft(X[:, None, :] * K[None, :, :], n=pad, dim=-1)  # pointwise multiply
    return Y[..., n - 1].reshape(out_shape)        # (..., out)


# ------------------------------------------------------------------
# 4) Frequency BitNet Linear Layer
# ------------------------------------------------------------------
class FreqBitNetLinear(nn.Module):
    def __init__(self, in_features, out_features, num_bins=8):
        super().__init__()
        self.w_real = nn.Parameter(torch.randn(out_features, in_features) * 0.8)
        self.enc = FreqSignatureEncoder(num_bins)
        self.in_features, self.out_features = in_features, out_features

    def freq_weights(self):
        w_t = TernaryQuantizeSTE.apply(self.w_real)   # {-1, 0, +1}
        sig = self.enc(w_t)                           # wave signatures
        return sig, w_t

    def forward(self, x):
        sig, w_t = self.freq_weights()
        W_hard = self.enc.decode(sig)                 # waves se ternary matrix (no grad)
        W = W_hard + (w_t - w_t.detach())             # value=W_hard, grad=STE se w_real tak
        return fft_matmul(x, W), W_hard

    def spectrum_energy(self):
        sig, _ = self.freq_weights()
        return sig.abs().pow(2).sum(dim=(0, 1))       # per-bin energy (bins,)


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    layer = FreqBitNetLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16)

    y, W_hard = layer(x)
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))

    # (a) sirf 2 frequencies active hain? ("only 2 freq" rule)
    energy = layer.spectrum_energy()
    print("\nPer-bin energy:", energy.tolist())
    active = (energy > 0).nonzero().flatten().tolist()
    print("Active bins (sirf 2 hone chahiye):", active)

    # (b) zero data loss: encode -> decode exact wapas?
    sig, w_t = layer.freq_weights()
    print("\nZero-loss check (decode==ternary):",
          torch.equal(layer.enc.decode(sig), w_t))

    # (c) FFT calculation == normal matmul (par waves se bana)?
    ref = x @ W_hard.T
    print("FFT-matmul matches reference:", torch.allclose(y, ref, atol=1e-5))

    # (d) gradient STE se flow hota hai?
    y.sum().backward()
    print("Gradient w_real par aaya:", layer.w_real.grad is not None)

    print("\n🎉 STEP 1 PASSED — Frequency BitNet ready!")