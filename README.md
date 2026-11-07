# 🌊 non_freq — Wave-Native AI Architecture

> **Numbers nahi, WAVES.**  
> Ek aisa AI blueprint jisme core mein koi floating-point numbers nahi, koi tokenizer nahi, aur koi matrix-multiply (GEMM) nahi.  
> Computation = **Wave Interference**. Data travel = **Soliton Waves (Zero Loss)**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-purple.svg)]()

---

## 🧠 The Philosophy

Aaj ke saare LLMs (GPT, Llama, Gemini) **numbers ke pinjre** mein qaid hain:
1. Text ko pehle **tokens** mein todo (BPE/SentencePiece).
2. Tokens ko **floating-point vectors** banao.
3. Unhe **matrix-multiply (GEMM)** karke GPU par megawatts bijli jalao.

**`non_freq` is paradigm ko todta hai.**  
Yahan data **waves** hai, weights **frequencies** hain, aur calculation **FFT (Fast Fourier Transform)** ke zariye hoti hai. Ye architecture human brain aur physics ke zyada kareeb hai — jahan information spikes/waves ke roop mein travel karti hai, decimal numbers ke roop mein nahi.

---

## ✨ Khaasiyat (Key Features)

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
| 6 | `step6_freq_back.py` | Frequency backpropagation |
| 7 | `step7_training_loop.py` | Tokenizer-free training loop |
| 8 | `step8_loop_thinking.py` | Loop thinking (ACT) |
| 9 | `step9_unified.py` | Unified model + eval + export |

Run: `python step9_unified.py`

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
