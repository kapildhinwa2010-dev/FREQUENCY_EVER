"""Bounded smoke tests - har step ka core path crash-free hai (no heavy training)."""
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
    ch = SolitonChannel()
    x = torch.randn(2, 24, 16)
    y = ch(x, 2.0)
    assert y.shape == x.shape
    assert (y - x).abs().max().item() < 0.05
    rep = ch.audit(x, 2.0)
    assert rep["zero_loss_err"] < 0.05
    assert rep["energy_ratio"] > 0.9


def test_soliton_channel_nD_grad_safe():
    """scalar/1D/2D grads (soliton_grad_pass inume inhi pe chalta hai)"""
    ch = SolitonChannel()
    for shape in [(), (7,), (3, 5), (2, 6, 4)]:
        g = torch.randn(shape)
        y = ch(g, 1.0)
        assert y.shape == g.shape
        assert (y - g).abs().max().item() < 0.05


def test_soliton_proof_shape_preserved():
    cs, cl = soliton_proof(T=128, T0=8.0, distance=16.0, steps=16)
    assert cs > 0.95, f"soliton shape not preserved: {cs}"


def test_unified_model_forward_and_generate():
    torch.manual_seed(0)
    model = NonFreqModel(d_model=16, n_layers=1, max_loops=2)
    b, _, waves = model.conv.encode("abcdefgh" * 4)
    lg = model(waves.unsqueeze(0))
    assert lg.shape[-1] == 256
    assert lg.shape[-2] == b.shape[0]
    out = model.generate("abc", 4)
    assert isinstance(out, str)
