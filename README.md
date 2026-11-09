# 🌊 non_freq — Wave-Native AI Architecture

> **Not numbers — WAVES.**  
> An AI blueprint whose core has no floating-point numbers, no tokenizer, and no matrix multiplication (GEMM).  
> Computation = **Wave Interference**. Data travel = **Soliton Waves (Zero Loss)**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-purple.svg)]()
[![CI run-all-steps](https://github.com/kapildhinwa2010-dev/FREQUENCY_EVER/actions/workflows/run-all-steps.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FREQUENCY_EVER/actions/workflows/run-all-steps.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/FREQUENCY_EVER/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/FREQUENCY_EVER/actions/workflows/codeql.yml)

---

## 🧠 The Philosophy

Every LLM today (GPT, Llama, Gemini) is **locked inside a cage of numbers**:
1. First, break text into **tokens** (BPE/SentencePiece).
2. Turn the tokens into **floating-point vectors**.
3. **Matrix-multiply (GEMM)** them and burn megawatts of electricity on GPUs.

**`non_freq` breaks this paradigm.**  
Here, data is **waves**, weights are **frequencies**, and computation happens via the **FFT (Fast Fourier Transform)**. This architecture is closer to the human brain and to physics — where information travels as spikes/waves, not as decimal numbers.

---

## ✨ Key Features

| Feature | Traditional LLMs | `non_freq` 🌊 |
| :--- | :--- | :--- |
| **Input Processing** | Tokenizer (BPE/WordPiece) | ❌ **Tokenizer-Free** (Raw UTF-8 Bytes → Waves) |
| **Vocabulary** | 32k - 100k+ tokens | **256 Bytes** (Language & Modality Agnostic) |
| **Core Math** | Matrix Multiplication (GEMM) | ❌ **Zero GEMM** (FFT Pointwise Multiply) |
| **Weights** | FP16/FP32 / INT8 | **Ternary {-1, 0, +1} as Frequencies** |
| **Data Transfer** | Numeric Tensors | **Soliton Waves** (Unitary ⇒ Zero Data Loss) |
| **Reasoning** | Feed-forward / Attention | **Loop Thinking** (Adaptive Computation Time) |
| **Backpropagation**| Numeric Chain Rule | **Frequency-Domain Chain Rule** |
| **Dependencies** | Heavy (Transformers, mamba-ssm) | **Pure PyTorch** (Zero external AI libs) |

---

## 🗂️ Repository Structure

| Step | File | Module |
|---|---|---|
| 1 | `step1_freq_bitnet.py` | Ternary weights as frequencies |
| 2 | `step2_converter.py` | Internal converter (no tokenizer) |
| 3 | `step3_mambabyte_freq.py` | MambaByte + frequency bytes |
| 4 | `step4_soliton_transfer.py` | REAL NLSE soliton channels (SSFM, N=1) |
| 5 | `step5_mamba2_fft.py` | Mamba-2 + FFT calc engine |
| 6 | `step6_freq_backprop.py` | Frequency backpropagation |
| 7 | `step7_training_loop.py` | Tokenizer-free training loop |
| 8 | `step8_loop_thinking.py` | Loop thinking (ACT) |
| 9 | `step9_unified.py` | Unified model + eval + export |

Run: `python step9_unified.py`

---

## 📦 Installation

```bash
pip install .            # runtime
pip install -e ".[dev]"   # dev (pytest)
```

## 🧪 Tests & CI

```bash
python -m pytest tests/ -q    # bounded smoke tests (seconds me)
```

- **Bounded smoke tests**: converter roundtrip, soliton zero-loss physics (SSFM), 0D-3D gradient safety, soliton shape proof, unified model forward+generate
- **run-all-steps CI**: relevant pushes pe 9 step demos, `ci_logs/` workflow artifact ke roop me upload hote hain
- **CodeQL** weekly security scan + Dependabot

## ⚠️ Research Status (Honest Note)

Ye ek **research prototype** hai. Physics assertions experimental smoke checks hain, broad validation nahi;
checked-in `ci_logs/pytest.log` mein failures recorded hain, isliye isse passing ya current verification na samjhein.
architecture kisi production LLM ka replacement nahi hai. Bina proof ke koi accuracy/benchmark claim nahi kiya gaya.

---

## 🏗️ Architecture & Data Flow

```text
[ Raw Text / Bytes / Any Data ]
            │
            ▼  (step2_converter.py — NO TOKENIZER)
[ Bit-Carrier Waves: (T, 8 carriers) ]
            │
            ▼  (FreqByteFrontend)
[ Hidden Stream ] ──── 🌊 Soliton Link (Zero-Loss) ────┐
            │                                          │
            ▼                                          │
[ Mamba-2 FFT Block (SSD via FFT) ]                    │
            │                                          │
            └──── 🌊 Soliton Link ◄────────────────────┘  (× N Layers)
            │
            ▼  🌊 Soliton Link
[ 🧠 Loop Thinking Module ]
  ├─ Weight-Tied Thought Loops
  └─ ACT Halt Gate (Adaptive Reasoning)
            │
            ▼
[ Ternary Freq-Head (Weights = 2 Carriers + Silence) ]
            │
            ▼
[ Next-Byte Logits (256) ] ──► Output
