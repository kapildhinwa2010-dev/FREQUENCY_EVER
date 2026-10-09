"""Utilities for converting raw text/tokens into wave representations."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

BITS = 8
BYTES_PER_TOKEN = 2


def text_to_bytes(text: str) -> torch.Tensor:
    """Convert UTF-8 text into a byte tensor."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)}")
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def tokens_to_bytes(tokens: Sequence[int] | Iterable[int] | torch.Tensor) -> torch.Tensor:
    """Convert integer token ids into a byte stream using 2 bytes per token."""
    if tokens is None:
        return torch.tensor([], dtype=torch.uint8)

    tensor = torch.as_tensor(tokens, dtype=torch.int64).flatten()
    out: list[int] = []
    for tok in tensor.tolist():
        value = int(tok)
        value = max(0, min(value, 2 ** (8 * BYTES_PER_TOKEN) - 1))
        out.extend(value.to_bytes(BYTES_PER_TOKEN, byteorder="little", signed=False))
    return torch.tensor(out, dtype=torch.uint8)


def bytes_to_bits(b: torch.Tensor) -> torch.Tensor:
    """Convert uint8 bytes into 8 binary bit channels."""
    if b.dim() == 0:
        b = b.unsqueeze(0)
    if b.dtype != torch.uint8:
        b = b.to(torch.uint8)
    shifts = torch.arange(BITS, device=b.device, dtype=torch.int64)[None, :]
    return ((b.to(torch.int64)[:, None] >> shifts) & 1).float()


def bits_to_bytes(bits: torch.Tensor) -> torch.Tensor:
    """Convert [T, 8] bits back into uint8 bytes."""
    if bits.dim() == 1:
        bits = bits.unsqueeze(0)
    if bits.dtype not in (torch.float32, torch.float64):
        bits = bits.float()
    weights = (2 ** torch.arange(BITS, device=bits.device, dtype=bits.dtype))
    out = (bits @ weights).to(torch.uint8)
    return out.squeeze(0) if out.shape[0] == 1 else out


class ByteToWave(nn.Module):
    """Map bit vectors to complex waves with a phase carrier structure."""

    def __init__(self, num_carriers: int = BITS):
        super().__init__()
        self.num_carriers = int(num_carriers)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.dim() != 2:
            raise ValueError(f"bits must have shape [T, 8], got {tuple(bits.shape)}")
        T = bits.shape[0]
        t = torch.arange(T, device=bits.device, dtype=torch.float32)[:, None]
        k = torch.arange(1, self.num_carriers + 1, device=bits.device, dtype=torch.float32)[None, :]
        phase = 2.0 * torch.pi * k * t / max(T, 1)
        return bits.to(torch.cfloat) * torch.exp(1j * phase)

    def decode_bits(self, waves: torch.Tensor) -> torch.Tensor:
        if waves.dim() != 2:
            raise ValueError(f"waves must have shape [T, 8], got {tuple(waves.shape)}")
        return (waves.abs() > 0.5).float()


class InternalConverter(nn.Module):
    """Encode raw text/tokens into byte-bit-wave representations."""

    def __init__(self):
        super().__init__()
        self.wave = ByteToWave(BITS)

    @staticmethod
    def _as_byte_tensor(x) -> torch.Tensor:
        if isinstance(x, str):
            return text_to_bytes(x)
        if torch.is_tensor(x):
            if x.dtype == torch.uint8:
                return x
            if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                return tokens_to_bytes(x)
            return tokens_to_bytes(x.tolist())
        return tokens_to_bytes(x)

    def encode(self, x):
        b = self._as_byte_tensor(x)
        bits = bytes_to_bits(b)
        waves = self.wave(bits)
        return b, bits, waves

    def decode(self, waves: torch.Tensor) -> torch.Tensor:
        bits = self.wave.decode_bits(waves)
        return bits_to_bytes(bits)

    def decode_text(self, waves: torch.Tensor) -> str:
        out = self.decode(waves).tolist()
        return bytes(out).decode("utf-8", errors="replace")


__all__ = [
    "BITS",
    "BYTES_PER_TOKEN",
    "text_to_bytes",
    "tokens_to_bytes",
    "bytes_to_bits",
    "bits_to_bytes",
    "ByteToWave",
    "InternalConverter",
]
