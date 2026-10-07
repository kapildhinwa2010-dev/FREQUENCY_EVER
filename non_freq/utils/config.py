from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import List

@dataclass
class SolitonConfig:
    enabled: bool = True
    zero_loss: bool = True
    carrier_freq: float = 1.0
    amplitude: float = 1.0

@dataclass
class FreqConfig:
    fft_type: str = "rfft"
    num_freq_bins: int = 64
    use_only_2_freq: bool = False

@dataclass
class BitNetConfig:
    enabled: bool = True
    levels: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    scale: float = 1.0

@dataclass
class MambaConfig:
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    byte_vocab: int = 256

@dataclass
class TrainingConfig:
    batch_size: int = 8
    seq_len: int = 64
    lr: float = 1e-3
    steps: int = 100

@dataclass
class Config:
    soliton: SolitonConfig = field(default_factory=SolitonConfig)
    freq: FreqConfig = field(default_factory=FreqConfig)
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        soliton=SolitonConfig(**raw.get("soliton", {})),
        freq=FreqConfig(**raw.get("freq", {})),
        bitnet=BitNetConfig(**raw.get("bitnet", {})),
        mamba=MambaConfig(**raw.get("mamba", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )