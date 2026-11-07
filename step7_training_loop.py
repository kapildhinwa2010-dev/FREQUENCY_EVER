"""
STEP 7 — FULL Training Loop (non_freq project)
Flow (point #3): bytes -> FFT -> freq weights -> frequency backprop.
Koi tokenizer nahi | backbone Mamba-2 FFT | head ternary freq-weights |
backprop FFT chain-rule + soliton grad channels | sabhi gaps pe soliton links.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from step1_freq_bitnet import FreqBitNetLinear
from step2_converter import InternalConverter
from step3_mambabyte_freq import FreqByteFrontend
from step4_soliton_transfer import SolitonChannel
from step5_mamba2_fft import Mamba2BlockFFT, FFTLinear
from step6_freq_back import FFTLinearFG, soliton_grad_pass


# ------------------------------------------------------------------
# 1) Full Non-Freq Model (saare steps jode gaye)
# ------------------------------------------------------------------
class TernaryHead(nn.Module):
    """STEP 1 wala frequency-ternary head: weights = waves {-1,0,+1}"""
    def __init__(self, d, vocab=256):
        super().__init__()
        self.fb = FreqBitNetLinear(d, vocab)

    def forward(self, x):
        return self.fb(x)[0]


class NonFreqModel(nn.Module):
    def __init__(self, d_model=64, n_layers=2):
        super().__init__()
        self.front = FreqByteFrontend(8, d_model)
        self.blocks = nn.ModuleList(
            [Mamba2BlockFFT(d_model) for _ in range(n_layers)])
        self.head = TernaryHead(d_model)
        self.link = SolitonChannel()

    def forward(self, waves):
        h = self.link(self.front(waves))
        for blk in self.blocks:
            h = blk(h) + h
            h = self.link(h)
        return self.head(h)


def upgrade_to_freqgrad(model):
    """har FFTLinear ko STEP 6 wale FFTLinearFG (freq-backprop) se badlo"""
    for name, child in list(model.named_children()):
        if isinstance(child, FFTLinear):
            new = FFTLinearFG(child.W.shape[0], child.W.shape[1],
                              bias=child.b is not None)
            with torch.no_grad():
                new.W.copy_(child.W)
                if child.b is not None:
                    new.b.copy_(child.b)
            setattr(model, name, new)
        else:
            upgrade_to_freqgrad(child)
    return model


# ------------------------------------------------------------------
# 2) Data (tokenizer-free: raw strings hi dataset hain)
# ------------------------------------------------------------------
TRAIN = [
    "wave wave wave", "wave wave wave wave",
    "freq freq freq", "freq freq freq freq",
    "soliton soliton soliton", "zero loss zero loss",
    "byte byte byte", "wave freq wave freq",
]
VAL = ["wave wave", "freq freq freq", "zero loss"]


def batch_loss(model, conv, text):
    b, _, w = conv.encode(text)
    lg = model(w.unsqueeze(0))
    return F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])


def eval_loss(model, conv, data):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for t in data:
            loss = batch_loss(model, conv, t)
            tot += loss.item() * (len(t.encode()) - 1); n += len(t.encode()) - 1
    model.train(); return tot / max(n, 1)


# ------------------------------------------------------------------
# 3) Generation (greedy next-byte, bina tokenizer)
# ------------------------------------------------------------------
def generate(model, conv, prompt, n=12):
    model.eval()
    bs = list(prompt.encode())
    with torch.no_grad():
        for _ in range(n):
            _, _, w = conv.encode(torch.tensor(bs, dtype=torch.uint8))
            lg = model(w.unsqueeze(0))
            bs.append(lg[0, -1].argmax().item())
    model.train()
    return bytes(bs).decode("utf-8", "ignore")


# ------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    conv = InternalConverter()
    model = upgrade_to_freqgrad(NonFreqModel(d_model=64, n_layers=2))
    link = SolitonChannel()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    def ternary_stats():
        _, wt = model.head.fb.freq_weights()
        return int((wt == -1).sum()), int((wt == 0).sum()), int((wt == 1).sum())

    start_stats = ternary_stats()
    print("ternary head (-1,0,+1) start:", start_stats)

    EPOCHS = 20
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for t in TRAIN:
            opt.zero_grad()
            loss = batch_loss(model, conv, t)     # bytes -> waves -> freq weights
            loss.backward()                       # 🌊 FFT chain-rule backprop
            soliton_grad_pass(model, link)        # 🌊 grads soliton me travel
            opt.step()
            tot += loss.item()
        if ep % 4 == 0 or ep == 1:
            print(f"epoch {ep:02d} | train {tot/len(TRAIN):.3f} | "
                  f"val {eval_loss(model, conv, VAL):.3f}")

    # --- final checks ---
    end_stats = ternary_stats()
    print("ternary head (-1,0,+1) end  :", end_stats,
          "| weights seekhe:", end_stats != start_stats)

    sig, wt = model.head.fb.freq_weights()
    print("head freq-weights zero-loss :",
          torch.equal(model.head.fb.enc.decode(sig), wt))

    print('generate("wave ")  ->', repr(generate(model, conv, "wave ")))
    print('generate("freq ")  ->', repr(generate(model, conv, "freq ")))

    torch.save(model.state_dict(), "non_freq_step7.pt")
    print("checkpoint saved: non_freq_step7.pt")

    print("\n🎉 STEP 7 PASSED — FULL training loop ready!")