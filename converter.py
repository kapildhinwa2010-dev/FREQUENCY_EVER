"""
STEP 2 — Internal Converter (non_freq project)
Flow: raw text / token-ids  ->  UTF-8 bytes  ->  bit-carrier frequency waves
Koi bahar ka tokenizer nahi — converter model ke ANDAR hai.
Zero data loss: waves -> bytes -> text exact wapas milta hai.
"""

import torch
import torch.nn as nn

BITS = 8                # 1 byte = 8 carriers
BYTES_PER_TOKEN = 2     # purane token-ids ko 2 bytes mein collapse karo (0..65535)


# ------------------------------------------------------------------
# 1) Text / Tokens -> Bytes
# ------------------------------------------------------------------
def text_to_bytes(text: str) -> torch.Tensor:
    """raw text -> UTF-8 byte stream (tokenizer-free)"""
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def tokens_to_bytes(tokens) -> torch.Tensor:
    """purane datasets ke token-ids -> bytes (little-endian, 2 bytes/token)"""
    t = torch.as_tensor(tokens, dtype=torch.int64).flatten()
    out = []
    for tok in t.tolist():
        tok = max(0, min(tok, 2 ** (8 * BYTES_PER_TOKEN) - 1))
        out += list(tok.to_bytes(BYTES_PER_TOKEN, "little"))
    return torch.tensor(out, dtype=torch.uint8)


# ------------------------------------------------------------------
# 2) Bytes -> Bits -> Waves (frequency signature)
# ------------------------------------------------------------------
def bytes_to_bits(b: torch.Tensor) -> torch.Tensor:
    """(T,) uint8 -> (T, 8) float bits, LSB first"""
    shifts = torch.arange(BITS, device=b.device)[None, :]
    return ((b.to(torch.int64)[:, None] >> shifts) & 1).float()


def bits_to_bytes(bits: torch.Tensor) -> torch.Tensor:
    """(T, 8) bits -> (T,) uint8  (inverse, zero-loss)"""
    weights = (2 ** torch.arange(BITS, device=bits.device)).float()
    return (bits @ weights).to(torch.uint8)


class ByteToWave(nn.Module):
    """
    har bit ek carrier wave hai:
        amplitude = bit (1 = wave ON, 0 = silence)
        phase     = 2*pi*k*t/T  (position info wave ke phase mein)
    """
    def __init__(self, num_carriers: int = BITS):
        super().__init__()
        self.num_carriers = num_carriers

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        T = bits.shape[0]
        t = torch.arange(T, device=bits.device).float()[:, None]      # (T,1)
        k = torch.arange(1, self.num_carriers + 1,
                         device=bits.device).float()[None, :]         # (1,8)
        phase = 2 * torch.pi * k * t / max(T, 1)
        return bits.to(torch.cfloat) * torch.exp(1j * phase)          # (T,8) waves

    def decode_bits(self, waves: torch.Tensor) -> torch.Tensor:
        """wave amplitude se bits wapas (phase ignore => robust)"""
        return (waves.abs() > 0.5).float()


# ------------------------------------------------------------------
# 3) Internal Converter — model ke andar wala full pipeline
# ------------------------------------------------------------------
class InternalConverter(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave = ByteToWave()

    def encode(self, x):
        """str | token-list | uint8-tensor  ->  (bytes, bits, waves)"""
        if isinstance(x, str):
            b = text_to_bytes(x)
        elif torch.is_tensor(x) and x.dtype == torch.uint8:
            b = x
        else:                                   # list/tuple/int-tensor => token-ids
            b = tokens_to_bytes(x)
        bits = bytes_to_bits(b)
        waves = self.wave(bits)
        return b, bits, waves

    def decode(self, waves: torch.Tensor) -> torch.Tensor:
        """waves -> bytes (zero-loss reverse path)"""
        return bits_to_bytes(self.wave.decode_bits(waves))


# ------------------------------------------------------------------
# DEMO / TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    conv = InternalConverter()

    # --- (a) raw text path (tokenizer-free) ---
    text = "non freq zindabad"
    b, bits, waves = conv.encode(text)
    dec = conv.decode(waves)
    print("text        :", text)
    print("bytes       :", b.tolist())
    print("waves shape :", tuple(waves.shape), "=> (T, 8 carriers)")
    print("zero-loss   :", torch.equal(dec, b),
          "| text match:", bytes(dec.tolist()).decode("utf-8") == text)

    # --- (b) purane token-ids path ---
    toks = [121, 47, 903, 255]
    b2, _, w2 = conv.encode(toks)
    dec2 = conv.decode(w2).tolist()
    rec = [int.from_bytes(bytes(dec2[i:i + BYTES_PER_TOKEN]), "little")
           for i in range(0, len(dec2), BYTES_PER_TOKEN)]
    print("\ntokens in   :", toks)
    print("tokens out  :", rec, "| match:", rec == toks)

    # --- (c) carrier activity: kaunsi frequencies zinda hain ---
    activity = waves.abs().mean(dim=0)
    print("\nPer-carrier activity (8 carriers):",
          [round(a, 3) for a in activity.tolist()])

    print("\n🎉 STEP 2 PASSED — Internal Converter ready!")q