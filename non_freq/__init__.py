"""Public package interface for non_freq."""

from .converter import ByteToWave, InternalConverter, bits_to_bytes, bytes_to_bits, text_to_bytes, tokens_to_bytes
from .freq_bitnet import FreqBitNetLinear, FreqSignatureEncoder, TernaryQuantizeSTE
from .fft import FFTLinear, FFTLinearFG, FFTMatMulFn, Mamba2BlockFFT, fft_causal_conv, fft_matmul, fft_matmul_b
from .loop_thinking import HaltGate, LoopThinking, NonFreqThinker
from .mamba import ByteFrontend, FreqByteFrontend, MambaBlock, MambaByteFreq
from .soliton import SolitonChannel, UnitaryWaveLink, soliton_proof
from .unified import NonFreqModel, benchmark, evaluate, export, stability_audit

__all__ = [
    "ByteToWave",
    "InternalConverter",
    "text_to_bytes",
    "tokens_to_bytes",
    "bytes_to_bits",
    "bits_to_bytes",
    "TernaryQuantizeSTE",
    "FreqSignatureEncoder",
    "FreqBitNetLinear",
    "fft_matmul",
    "fft_matmul_b",
    "fft_causal_conv",
    "FFTLinear",
    "FFTLinearFG",
    "FFTMatMulFn",
    "Mamba2BlockFFT",
    "FreqByteFrontend",
    "ByteFrontend",
    "MambaBlock",
    "MambaByteFreq",
    "SolitonChannel",
    "UnitaryWaveLink",
    "soliton_proof",
    "HaltGate",
    "LoopThinking",
    "NonFreqThinker",
    "NonFreqModel",
    "evaluate",
    "stability_audit",
    "benchmark",
    "export",
]

__version__ = "0.1.0"
