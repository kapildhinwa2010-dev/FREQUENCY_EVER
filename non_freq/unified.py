"""Unified, ready-to-run model entry point for the project."""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from non_freq.converter import InternalConverter
from non_freq.fft import FFTLinear, FFTMatMulFn, Mamba2BlockFFT, fft_matmul, soliton_grad_pass
from non_freq.freq_bitnet import FreqSignatureEncoder, TernaryQuantizeSTE
from non_freq.loop_thinking import LoopThinking
from non_freq.soliton import SolitonChannel, soliton_proof


class FreqBitNetLinearFG(nn.Module):
    """A usable ternary FFT-based output head."""

    def __init__(self, in_features, out_features, num_bins=8):
        super().__init__()
        self.w_real = nn.Parameter(torch.randn(out_features, in_features) * 0.8)
        self.enc = FreqSignatureEncoder(num_bins)

    def ternary_weights(self):
        w_t = TernaryQuantizeSTE.apply(self.w_real)
        sig = self.enc(w_t)
        W_hard = self.enc.decode(sig).t()
        return W_hard + (w_t.t() - w_t.t().detach()), sig

    def forward(self, x):
        W, _ = self.ternary_weights()
        return FFTMatMulFn.apply(x, W)

    def spectrum_energy(self):
        _, sig = self.ternary_weights()
        return sig.abs().pow(2).sum(dim=(0, 1))


class NonFreqModel(nn.Module):
    """Unified model that accepts raw text or waves and returns logits."""

    def __init__(self, d_model=64, n_layers=2, max_loops=4, ternary_head=True, num_bins=8, **kw):
        super().__init__()
        self.conv = InternalConverter()
        self.front = nn.Linear(2 * 8, d_model)
        self.blocks = nn.ModuleList([Mamba2BlockFFT(d_model, **kw) for _ in range(n_layers)])
        self.link = SolitonChannel()
        self.loop = LoopThinking(d_model, max_loops=max_loops, **kw)
        self.head = FreqBitNetLinearFG(d_model, 256, num_bins) if ternary_head else FFTLinear(d_model, 256)
        self.cfg = {"d_model": d_model, "n_layers": n_layers, "max_loops": max_loops, "ternary_head": ternary_head}

    def to_waves(self, x):
        if torch.is_tensor(x) and x.dtype == torch.cfloat:
            return x
        return self.conv.encode(x)[2]

    def forward(self, x, return_thoughts=False):
        w = self.to_waves(x)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        h = self.link(self.front(torch.cat([w.real, w.imag], dim=-1)))
        for blk in self.blocks:
            h = self.link(blk(h) + h)
        h_th, ponder, trace = self.loop(h)
        lg = self.head(h_th)
        return (lg, ponder, trace) if return_thoughts else lg

    @torch.no_grad()
    def generate(self, seed: str, n: int = 20, ctx_len: int = 32):
        ctx = list(seed.encode("utf-8"))
        start = len(ctx)
        device = next(self.parameters()).device
        for _ in range(n):
            token_window = torch.tensor(ctx[-ctx_len:], dtype=torch.uint8, device=device)
            w = self.to_waves(token_window)
            if w.dim() == 2:
                w = w.unsqueeze(0)
            logits = self(w)
            next_idx = logits[0, -1].argmax().item()
            ctx.append(next_idx)
        return bytes(ctx[start:]).decode("utf-8", errors="ignore")


@torch.no_grad()
def evaluate(model, text):
    """Evaluate model on a text sample."""
    b, _, waves = model.conv.encode(text)
    lg, ponder, trace = model(waves.unsqueeze(0), return_thoughts=True)
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), b.long()[1:])
    acc = (lg[:, :-1].argmax(-1) == b.long()[1:].unsqueeze(0)).float().mean().item()
    return {
        "loss": round(float(loss.item()), 4),
        "acc": round(float(acc), 4),
        "avg_loops": round(float(ponder.mean().item()), 2),
        "thought_trace": [round(float(t), 3) for t in trace],
    }


@torch.no_grad()
def stability_audit(model):
    """Audit stability metrics using soliton physics."""
    rep = model.link.audit(torch.randn(2, 24, 32), 2.0)
    rep = {k: round(float(v), 6) for k, v in rep.items()}
    if isinstance(model.head, FreqBitNetLinearFG):
        e = model.head.spectrum_energy()
        rep["head_active_freq_bins"] = (e > 0).nonzero().flatten().tolist()
    return rep


def benchmark(sizes=(128, 256, 512), T=256, reps=20):
    """Benchmark FFT matmul vs standard matmul."""
    rows = []
    for n in sizes:
        x, W = torch.randn(T, n), torch.randn(n, n)
        t0 = time.perf_counter()
        for _ in range(reps):
            fft_matmul(x, W)
        tf = (time.perf_counter() - t0) / reps * 1e3

        t0 = time.perf_counter()
        for _ in range(reps):
            x @ W
        tm = (time.perf_counter() - t0) / reps * 1e3
        rows.append((n, round(float(tf), 3), round(float(tm), 3)))
    return rows


def export(model, path="nonfreq_final.pt"):
    """Export model state and project blueprint."""
    torch.save({"state": model.state_dict(), "cfg": model.cfg}, path)
    blueprint = {
        "project": "non_freq",
        "philosophy": "Numbers nahi, WAVES — computation = wave interference.",
        "rules": [
            "weights: ternary {-1,0,+1}",
            "input: no tokenizer; text -> bytes -> bit-carrier waves",
            "calc: FFT / wave-domain operations",
            "transfer: soliton-inspired channels",
            "backprop: autograd + soliton-style gradient pass",
        ],
        "config": model.cfg,
    }
    with open("blueprint.json", "w", encoding="utf-8") as f:
        json.dump(blueprint, f, indent=2, ensure_ascii=False)


README = """# non_freq 🌊

A clean, runnable wave-native architecture prototype.

## Quick start

```bash
pip install -r requirements.txt
python -m non_freq.unified
```

## Architecture

- **Input**: Raw text (no tokenizer) → UTF-8 bytes → bit-carrier waves
- **Core**: Mamba-2 style FFT blocks with soliton transfer channels
- **Thinking**: Loop-based reasoning with adaptive halt gates
- **Output**: Ternary frequency-weighted head

## Modules

- `converter.py`: text/token → bytes → waves
- `freq_bitnet.py`: ternary frequency weights
- `fft.py`: FFT-based linear ops and Mamba-2 blocks
- `soliton.py`: soliton-inspired transfer channel
- `loop_thinking.py`: adaptive loop thinking (ACT-style)
- `unified.py`: full model and utilities

## Run

```python
from non_freq.unified import NonFreqModel, evaluate, export

model = NonFreqModel(d_model=32, n_layers=1)
result = evaluate(model, "hello world")
export(model)
```
"""


def write_readme(path="README.md"):
    """Write README to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(README)


if __name__ == "__main__":
    print("🌊 NON_FREQ — Wave-Native AI Architecture")
    print("=" * 60)

    torch.manual_seed(1)
    print("\n[1] Initializing model...")
    model = NonFreqModel(d_model=32, n_layers=1, max_loops=4)

    print("[2] Quick training (20 steps)...")
    text = "soliton waves zero loss " * 20
    b, _, w = model.conv.encode(text)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(20):
        opt.zero_grad()
        logits = model(w.unsqueeze(0))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), b.long()[1:])
        loss.backward()
        soliton_grad_pass(model, model.link)
        opt.step()
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print("\n[3] Evaluation...")
    eval_result = evaluate(model, "soliton waves zero loss soliton waves")
    for key, val in eval_result.items():
        print(f"  {key}: {val}")

    print("\n[4] Physics stability audit...")
    audit = stability_audit(model)
    for key, val in audit.items():
        print(f"  {key}: {val}")

    print("\n[5] Soliton proof...")
    soliton_corr, linear_corr = soliton_proof()
    print(f"  Soliton shape preservation: {soliton_corr:.4f}")
    print(f"  Linear dispersion: {linear_corr:.4f}")

    print("\n[6] Benchmark (FFT vs matmul)...")
    benchmark_results = benchmark(sizes=(64, 128), T=256, reps=10)
    print("  (size, fft_ms, matmul_ms)")
    for row in benchmark_results:
        print(f"  {row}")

    print("\n[7] Text generation...")
    generated = model.generate("waves ", 15)
    print(f"  Generated: {repr(generated)}")

    print("\n[8] Export...")
    export(model)
    write_readme()
    print("  ✓ nonfreq_final.pt")
    print("  ✓ blueprint.json")
    print("  ✓ README.md")

    print("\n" + "=" * 60)
    print("🏁 NON_FREQ PROJECT COMPLETE!")
    print("🌊 DATA TRANSFER: Wave interference in frequency domain")
    print("=" * 60)
