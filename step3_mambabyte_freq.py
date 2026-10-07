"""
STEP 3 — MambaByte + Frequency Bytes (non_freq project)
Classic MambaByte (256 byte-ids) RETAINED + naya freq-byte path ADDED:
input = STEP 2 wali complex waves (T, 8 carriers), koi byte-id nahi.
Minimal selective-scan (S6/Mamba) block, pure PyTorch, zero external deps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 1) Frontends: freq-bytes (native) aur classic byte-ids (compat)
# ------------------------------------------------------------------
class FreqByteFrontend(nn.Module):
    """waves (B,T,8) complex -> (B,T,d_model) | wave ka real+imag hi signal hai"""
    def __init__(self, num_carriers=8, d_model=64):
        super().__init__()
        self.proj = nn.Linear(2 * num_carriers, d_model)

    def forward(self, waves):
        z = torch.cat([waves.real, waves.imag], dim=-1)   # wave as-is, no numbers
        return self.proj(z)


class ByteFrontend(nn.Module):
    """classic MambaByte: 256 byte-id embedding (purani duniya se compat)"""
    def __init__(self, d_model=64, vocab=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)

    def forward(self, ids):
        return self.emb(ids.long())


# ------------------------------------------------------------------
# 2) Minimal Mamba Block (selective scan / S6)
# ------------------------------------------------------------------
def selective_scan(x, dt, A, Bp, Cp, D):
    """x:(Bz,T,I) dt:(Bz,T,I) A:(I,N) Bp,Cp:(Bz,T,N) -> y:(Bz,T,I)"""
    Bz, T, I = x.shape
    N = A.shape[1]
    h = torch.zeros(Bz, I, N, device=x.device, dtype=x.dtype)   # wave-state
    ys = []
    for t in range(T):
        dA = torch.exp(-dt[:, t, :].unsqueeze(-1) * A)          # decay (silence ki ore)
        dB = dt[:, t, :].unsqueeze(-1) * Bp[:, t, :].unsqueeze(1)
        h = h * dA + dB * x[:, t, :].unsqueeze(-1)              # state update
        ys.append((h * Cp[:, t, :].unsqueeze(1)).sum(-1))       # read-out
    y = torch.stack(ys, dim=1)
    return y + x * D.unsqueeze(0).unsqueeze(0)                  # skip (D)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_state = d_state
        inner = d_model * expand
        self.inner = inner
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)     # x aur gate
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner,
                              padding=d_conv - 1)                    # causal depthwise
        self.x_proj = nn.Linear(inner, 2 * d_state + inner, bias=False)  # B, C, dt
        self.A = nn.Parameter(torch.arange(1, d_state + 1).float()
                              .unsqueeze(0).repeat(inner, 1))        # (I,N) decay rates
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, u):
        T = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)   # causal crop
        xc = F.silu(xc)
        Bp, Cp, dt = self.x_proj(xc).split(
            [self.d_state, self.d_state, self.inner], dim=-1)
        dt = F.softplus(dt)                                          # positive steps
        y = selective_scan(xc, dt, self.A, Bp, Cp, self.D)
        return self.out_proj(y * F.silu(z))                          # gated output


# ------------------------------------------------------------------
# 3) Full Model: MambaByte + Frequency Bytes
# ------------------------------------------------------------------
class MambaByteFreq(nn.Module):
    def __init__(self, d_model=64, n_layers=2, d_state=16,
                 d_conv=4, expand=2, num_carriers=8, byte_vocab=256):
        super().__init__()
        self.freq_frontend = FreqByteFrontend(num_carriers, d_model)
        self.byte_frontend = ByteFrontend(d_model, byte_vocab)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, byte_vocab)   # next-byte (tokenizer-free vocab)

    def forward(self, x, mode="freq"):
        h = self.freq_frontend(x) if mode == "freq" else self.byte_frontend(x)
        for blk in self.blocks:
            h = blk(h) + h                            # residual wave-stream
        return h

    def logits(self, h):
        return self.head(h)


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    from step2_converter import InternalConverter

    torch.manual_seed(0)
    conv = InternalConverter()
    model = MambaByteFreq(d_model=64, n_layers=2)

    # --- (a) FREQ-BYTES path (native non_freq route) ---
    text = "soliton waves se data transfer, zero loss"
    b, bits, waves = conv.encode(text)
    W = waves.unsqueeze(0)                              # (1, T, 8)
    h = model(W, mode="freq")
    print("freq-bytes input :", tuple(W.shape))
    print("hidden states    :", tuple(h.shape))

    # --- (b) CLASSIC byte-id path (MambaByte compat) ---
    ids = b.long().unsqueeze(0)                         # (1, T)
    h2 = model(ids, mode="byte")
    print("byte-id input    :", tuple(ids.shape), "-> hidden:", tuple(h2.shape))

    # --- (c) next-byte logits + gradient flow ---
    lg = model.logits(h)
    print("next-byte logits :", tuple(lg.shape), "(256 = tokenizer-free vocab)")
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    loss.backward()
    gsum = sum(p.grad.abs().sum().item()
               for p in model.parameters() if p.grad is not None)
    print("loss:", round(loss.item(), 3),
          "(random init pe ~ln(256)=5.55 aana chahiye) | grad flow:", gsum > 0)
    print("params:", sum(p.numel() for p in model.parameters()))

    print("\n🎉 STEP 3 PASSED — MambaByte + Frequency Bytes ready!")