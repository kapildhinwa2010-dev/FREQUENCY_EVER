"""Bounded smoke tests - har step ka core path crash-free hai (no heavy training).
Tolerances physics ke hisaab se: mean STRICT (<0.05), max lenient (<0.25)
kyunki SSFM discretization + tanh phase->value mapping edge values pe
thodi error amplify karti hai. Deterministic seeds ke saath."""
import torch

from step2_converter import InternalConverter
from step4_soliton_transfer import SolitonChannel, soliton_proof
from step9_unified import NonFreqModel


def test_converter_roundtrip():
    conv = InternalConverter()
    text = "non freq waves zindabad"
    b, bits, waves = conv.encode(text)
    assert torch.equal(conv.decode(waves), b)


def test_soliton_channel_zero_loss():
    torch.manual_seed(0)
    ch = SolitonChannel()
    x = torch.randn(2, 24, 16)
    y = ch(x, 2.0)
    assert y.shape == x.shape
    err = (y - x).abs()
    assert err.mean().item() < 0.05, f"mean err {err.mean().item():.4f}"
    assert err.max().item() < 0.25, f"max err {err.max().item():.4f} (tanh edge + SSFM)"
    rep = ch.audit(x, 2.0)
    assert rep["zero_loss_err"] < 0.25, rep
    assert rep["energy_ratio"] > 0.9, rep


def test_soliton_channel_nD_grad_safe():
    """scalar/1D/2D grads (soliton_grad_pass inhi pe chalta hai)"""
    torch.manual_seed(1)
    ch = SolitonChannel()
    for shape in [(), (7,), (3, 5), (2, 6, 4)]:
        g = torch.randn(shape)
        y = ch(g, 1.0)
        assert y.shape == g.shape
        err = (y - g).abs()
        assert err.mean().item() < 0.05, f"{shape} mean err {err.mean().item():.4f}"
        assert err.max().item() < 0.25, f"{shape} max err {err.max().item():.4f}"


def test_soliton_proof_shape_preserved():
    cs, cl = soliton_proof(T=128, T0=8.0, distance=16.0, steps=16)
    assert cs > 0.95, f"soliton shape not preserved: {cs}"
    assert cl < cs, "linear dispersion soliton se zyada preserve? impossible"


def test_unified_model_forward_and_generate():
    torch.manual_seed(0)
    model = NonFreqModel(d_model=16, n_layers=1, max_loops=2)
    b, _, waves = model.conv.encode("abcdefgh" * 4)
    lg = model(waves.unsqueeze(0))
    assert lg.shape[-1] == 256
    assert lg.shape[-2] == b.shape[0]
    out = model.generate("abc", 4)
    assert isinstance(out, str)
