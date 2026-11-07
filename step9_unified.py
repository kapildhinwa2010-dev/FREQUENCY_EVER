"""
STEP 9 (FINAL) — Unified NonFreqModel (non_freq project)
Imports = GitHub repo ke RENAMED file names (sahi naam):
  1. step1_freq_bitnet.py
  2. step2_converter.py
  3. step3_mambabyte_freq.py
  4. step4_soliton_transfer.py   <-- v2: REAL NLSE physics soliton channel
  5. step5_mamba2_fft.py
  6. step6_freq_backprop.py
  7. step7_training_loop.py
  8. step8_loop_thinking.py
Philosophy: numbers nahi, WAVES — computation = wave interference.
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RENAMED repo files (final sahi naam) ---
from step1_freq_bitnet import TernaryQuantizeSTE, FreqSignatureEncoder
from step2_converter import InternalConverter
from step3_mambabyte_freq import FreqByteFrontend
from step4_soliton_transfer import SolitonChannel, soliton_proof
from step5_mamba2_fft import Mamba2BlockFFT, FFTLinear, fft_matmul
from step6_freq_backprop import FFTMatMulFn, soliton_grad_pass
from step8_loop_thinking import LoopThinking


# ------------------------------------------------------------------
# 1) STEP1+5+6 UNION: ternary freq-weight + FFT calc + freq backprop
# ------------------------------------------------------------------
class FreqBitNetLinearFG(nn.Module):
    def __init__(self, in_f, out_f, num_bins=8):
        super().__init__()
        self.w_real = nn.Parameter(torch.randn(out_f, in_f) * 0.8)
        self.enc = FreqSignatureEncoder(num_bins)

    def ternary_weights(self):
        w_t = TernaryQuantizeSTE.apply(self.w_real)      # {-1,0,+1}
        sig = self.enc(w_t)                              # wave signatures
        W_hard = self.enc.decode(sig).t()                # (in,out)
        return W_hard + (w_t.t() - w_t.t().detach()), sig

    def forward(self, x):
        W, _ = self.ternary_weights()
        return FFTMatMulFn.apply(x, W)                   # 🌊 FFT calc + freq backprop

    def spectrum_energy(self):
        _, sig = self.ternary_weights()
        return sig.abs().pow(2).sum(dim=(0, 1))


# ------------------------------------------------------------------
# 2) UNIFIED MODEL
# ------------------------------------------------------------------
class NonFreqModel(nn.Module):
    def __init__(self, d_model=64, n_layers=2, max_loops=4,
                 ternary_head=True, num_bins=8, **kw):
        super().__init__()
        self.conv = InternalConverter()                  # step2_converter.py
        self.front = FreqByteFrontend(8, d_model)        # step3
        self.blocks = nn.ModuleList(
            [Mamba2BlockFFT(d_model, **kw) for _ in range(n_layers)])   # step5
        self.link = SolitonChannel()                     # step4 (REAL NLSE physics)
        self.loop = LoopThinking(d_model, max_loops, **kw)              # step8
        self.head = (FreqBitNetLinearFG(d_model, 256, num_bins)
                     if ternary_head else FFTLinear(d_model, 256))
        self.cfg = dict(d_model=d_model, n_layers=n_layers,
                        max_loops=max_loops, ternary_head=ternary_head)

    def to_waves(self, x):
        if torch.is_tensor(x) and x.dtype == torch.cfloat:
            return x
        return self.conv.encode(x)[2]                    # str/bytes/tokens -> waves

    def forward(self, x, return_thoughts=False):
        w = self.to_waves(x)
        if w.dim() == 2:                                 # string/2D input safety
            w = w.unsqueeze(0)
        h = self.link(self.front(w))
        for blk in self.blocks:
            h = self.link(blk(h) + h)
        h_th, ponder, trace = self.loop(h)               # 🧠 loop thinking
        lg = self.head(h_th)
        return (lg, ponder, trace) if return_thoughts else lg

    @torch.no_grad()
    def generate(self, seed: str, n: int = 20, ctx_len: int = 32):
        ctx = list(seed.encode("utf-8"))
        s0 = len(ctx)
        device = next(self.parameters()).device          # GPU/CPU safety
        for _ in range(n):
            t = torch.tensor(ctx[-ctx_len:], dtype=torch.uint8, device=device)
            w = self.to_waves(t)
            if w.dim() == 2:
                w = w.unsqueeze(0)
            ctx.append(self(w)[0, -1].argmax().item())
        return bytes(ctx[s0:]).decode("utf-8", "ignore")


# ------------------------------------------------------------------
# 3) EVAL LAYER + STABILITY AUDIT + BENCHMARK + EXPORT
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, text):
    b, _, waves = model.conv.encode(text)
    lg, ponder, trace = model(waves.unsqueeze(0), return_thoughts=True)
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    acc = (lg[:, :-1].argmax(-1) == b.long()[1:].unsqueeze(0)).float().mean().item()
    return dict(loss=round(loss.item(), 4), acc=round(acc, 4),
                avg_loops=round(ponder.mean().item(), 2),
                thought_trace=[round(t, 3) for t in trace])


def stability_audit(model):
    """REAL NLSE physics audit: zero-loss + soliton integrity + energy conservation"""
    rep = model.link.audit(torch.randn(2, 24, 32), 2.0)
    rep = {k: round(v, 6) for k, v in rep.items()}
    if isinstance(model.head, FreqBitNetLinearFG):
        e = model.head.spectrum_energy()
        rep["head_active_freq_bins"] = (e > 0).nonzero().flatten().tolist()
    return rep


def benchmark(sizes=(128, 256, 512), T=256, reps=20):
    rows = []
    for n in sizes:
        x, W = torch.randn(T, n), torch.randn(n, n)
        t0 = time.perf_counter()
        for _ in range(reps): fft_matmul(x, W)
        tf = (time.perf_counter() - t0) / reps * 1e3
        t0 = time.perf_counter()
        for _ in range(reps): x @ W
        tm = (time.perf_counter() - t0) / reps * 1e3
        rows.append((n, round(tf, 3), round(tm, 3)))
    return rows


def export(model, path="nonfreq_final.pt"):
    torch.save({"state": model.state_dict(), "cfg": model.cfg}, path)
    blueprint = {
        "project": "non_freq",
        "philosophy": "numbers nahi, waves — computation = wave interference",
        "rules": [
            "weights: ternary {-1,0,+1} = 2 carrier frequencies + silence",
            "input: no tokenizer; text -> bytes -> bit-carrier waves",
            "calc: FFT pointwise multiply (koi GEMM nahi)",
            "transfer: NLSE soliton channels (SSFM, N=1 phase-modulation => physics-grade zero loss)",
            "backprop: freq-domain chain rule + lossless identity grad through soliton",
            "thinking: weight-tied loops + ACT halt gate",
        ],
        "config": model.cfg,
        "repo_structure": [
            "step1_freq_bitnet.py",
            "step2_converter.py",
            "step3_mambabyte_freq.py",
            "step4_soliton_transfer.py",
            "step5_mamba2_fft.py",
            "step6_freq_backprop.py",
            "step7_training_loop.py",
            "step8_loop_thinking.py",
            "step9_unified.py",
        ],
    }
    with open("blueprint.json", "w") as f:
        json.dump(blueprint, f, indent=2, ensure_ascii=False)


README = """# non_freq 🌊
> Numbers nahi, WAVES. Computation = wave interference.

text -> bytes -> waves -> [Mamba-2 FFT blocks + NLSE soliton links] -> loop thinking -> next-byte

## Repository Structure
| Step | File | Module |
|---|---|---|
| 1 | `step1_freq_bitnet.py` | Ternary weights as frequencies |
| 2 | `step2_converter.py` | Internal converter (no tokenizer) |
| 3 | `step3_mambabyte_freq.py` | MambaByte + frequency bytes |
| 4 | `step4_soliton_transfer.py` | **REAL NLSE soliton channels (SSFM, N=1)** |
| 5 | `step5_mamba2_fft.py` | Mamba-2 + FFT calc engine |
| 6 | `step6_freq_backprop.py` | Frequency backpropagation |
| 7 | `step7_training_loop.py` | Tokenizer-free training loop |
| 8 | `step8_loop_thinking.py` | Loop thinking (ACT) |
| 9 | `step9_unified.py` | Unified model + eval + export |

Run: `python step9_unified.py`
"""


def write_readme():
    with open("README.md", "w") as f:
        f.write(README)


# ------------------------------------------------------------------
# FINAL DEMO
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1)
    model = NonFreqModel(d_model=32, n_layers=1, max_loops=4)

    # --- quick train (ternary freq-head ke saath) ---
    text = "soliton waves zero loss " * 20
    b, _, w = model.conv.encode(text)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(model(w.unsqueeze(0))[:, :-1].reshape(-1, 256),
                               b.long()[1:])
        loss.backward()
        soliton_grad_pass(model, model.link)             # 🌊 grads wave me
        opt.step()
    print("train loss end:", round(loss.item(), 3))

    # --- eval layer ---
    print("eval:", evaluate(model, "soliton waves zero loss soliton waves"))

    # --- REAL physics stability audit ---
    print("stability (NLSE physics):", stability_audit(model))

    # --- soliton proof: N=1 shape preserve vs linear dispersion ---
    cs, cl = soliton_proof()
    print(f"soliton proof | shape corr SOLITON: {cs:.4f} | linear: {cl:.4f}")

    # --- benchmark ---
    print("benchmark (n, fft_ms, matmul_ms):")
    for row in benchmark():
        print("   ", row)

    # --- generate ---
    print("generated:", repr(model.generate("soliton waves ", 20)))

    # --- export ---
    export(model)
    write_readme()
    print("exported: nonfreq_final.pt, blueprint.json, README.md")

    print("\n🏁 NON_FREQ PROJECT COMPLETE — notebook se code tak, saare 9 steps unified!")
    print("🌊 DATA TRANSFER: REAL NLSE SOLITON PHYSICS (SSFM, N=1 phase-modulation)")