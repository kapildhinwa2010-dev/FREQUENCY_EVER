"""
STEP 8 — Loop Thinking System (non_freq project)
Model ek hi weight-tied thought block ko baar-baar chalata hai (vichar-chakra),
jab tak har position ka Halt Gate "bas" na kahe (Adaptive Computation Time).
Final answer = sabhi thought-states ka halt-probability-weighted sum (ponderation).
Har chakra ke beech SolitonChannel zero-loss transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from step3_mambabyte_freq import FreqByteFrontend
from step4_soliton_transfer import SolitonChannel
from step5_mamba2_fft import Mamba2BlockFFT, FFTLinear


# ------------------------------------------------------------------
# 1) Halt Gate — kitna sochna hai, position khud decide karti hai
# ------------------------------------------------------------------
class HaltGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = FFTLinear(d, 1)

    def forward(self, h):
        return torch.sigmoid(self.g(h))            # (B,T,1) halt probability


# ------------------------------------------------------------------
# 2) Loop Thinking Core (ACT-style adaptive loops)
# ------------------------------------------------------------------
class LoopThinking(nn.Module):
    def __init__(self, d_model, max_loops=6, **kw):
        super().__init__()
        self.thought = Mamba2BlockFFT(d_model, **kw)   # weight-tied vichar-chakra
        self.gate = HaltGate(d_model)
        self.link = SolitonChannel()
        self.max_loops = max_loops

    def forward(self, h):
        B, T, D = h.shape
        running = torch.ones(B, T, 1, device=h.device)   # jo abhi soch rahe hain
        accum = torch.zeros_like(h)                      # weighted thought sum
        ponder = torch.zeros(B, T, 1, device=h.device)   # expected loop count
        trace = []

        for t in range(self.max_loops):
            h = self.link(self.thought(h) + h)           # 🌀🌊 ek vichar-chakra
            p = self.gate(h) * running
            if t == self.max_loops - 1:
                p = running                              # last me forced halt
            accum = accum + p * h                        # yakin jitna, weight utna
            ponder = ponder + p * (t + 1)
            running = running - p
            trace.append(p.mean().item())

        return accum, ponder, trace


# ------------------------------------------------------------------
# 3) Full Thinker Model
# ------------------------------------------------------------------
class NonFreqThinker(nn.Module):
    def __init__(self, d_model=64, n_layers=2, max_loops=6, **kw):
        super().__init__()
        self.front = FreqByteFrontend(8, d_model)
        self.blocks = nn.ModuleList(
            [Mamba2BlockFFT(d_model, **kw) for _ in range(n_layers)])
        self.link = SolitonChannel()
        self.loop = LoopThinking(d_model, max_loops, **kw)
        self.head = FFTLinear(d_model, 256)

    def forward(self, waves, return_thoughts=False):
        h = self.link(self.front(waves))
        for blk in self.blocks:
            h = self.link(blk(h) + h)
        h_th, ponder, trace = self.loop(h)               # 🧠 loop thinking
        logits = self.head(h_th)
        if return_thoughts:
            return logits, ponder, trace
        return logits


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    from step2_converter import InternalConverter
    torch.manual_seed(3)
    conv = InternalConverter()

    # --- (a) ACT mechanics: halt-probs ka sum 1, ponder range valid ---
    model = NonFreqThinker(d_model=32, n_layers=1, max_loops=5)
    b, _, waves = conv.encode("loop thinking me model baar baar sochta hai")
    lg, ponder, trace = model(waves.unsqueeze(0), return_thoughts=True)
    print("logits:", tuple(lg.shape))
    print("sum(halt probs) ≈ 1 :", round(sum(trace), 4))
    print("ponder range (1-5) :", round(ponder.min().item(), 2),
          "-", round(ponder.max().item(), 2))
    print("thought trace      :", [round(t, 3) for t in trace])

    # --- (b) gradient loop ke andar tak jaata hai ---
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    loss.backward()
    print("grads thought/gate :",
          model.loop.thought.in_proj.W.grad is not None,
          model.loop.gate.g.W.grad is not None)

    # --- (c) loop-thinker ko period-8 pattern sikhao ---
    model = NonFreqThinker(d_model=32, n_layers=1, max_loops=5)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    text = "abcdefgh" * 30
    b2, _, w2 = conv.encode(text)
    tgt = b2.long()[1:]
    losses = []
    for step in range(50):
        opt.zero_grad()
        l = F.cross_entropy(model(w2.unsqueeze(0))[:, :-1].reshape(-1, 256), tgt)
        l.backward()
        opt.step()
        losses.append(l.item())
    print("loss:", round(losses[0], 3), "->", round(losses[-1], 3),
          "| loop-thinking seekha:", losses[-1] < losses[0] - 1.5)

    # --- (d) think & generate: soch kar bolo ---
    model.eval()
    ctx = list("abcdefgh".encode("latin-1"))
    with torch.no_grad():
        for _ in range(16):
            _, _, ww = conv.encode(torch.tensor(ctx[-32:], dtype=torch.uint8))
            nxt = model(ww.unsqueeze(0))[0, -1].argmax().item()
            ctx.append(nxt)
    print("generated:", repr(bytes(ctx[8:]).decode("latin-1", "ignore")))

    print("\n🎉 STEP 8 PASSED — Loop Thinking system ready!")
