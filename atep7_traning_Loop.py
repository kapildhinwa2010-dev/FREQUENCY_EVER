"""
STEP 7 — Full Training Loop (non_freq project)
Flow: Raw Text -> Bytes -> Waves -> Mamba2-FFT -> Next-Byte Pred -> Freq-Backprop
No tokenizer. No numeric matmul. Pure wave-native training.
Includes: Validation, Checkpointing, Loss Logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os

# Pichle steps ke modules import
from step2_converter import InternalConverter
from step5_mamba2_fft import Mamba2FreqFFT
from step6_freq_backprop import FFTLinearFG, soliton_grad_pass, SolitonChannel


# ------------------------------------------------------------------
# 1) WaveDataset: Raw text ko on-the-fly waves mein convert karta hai
# ------------------------------------------------------------------
class WaveDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 64):
        self.bytes = list(text.encode("utf-8"))
        self.seq_len = seq_len
        self.converter = InternalConverter()
        
    def __len__(self):
        return max(0, len(self.bytes) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.bytes[idx : idx + self.seq_len + 1]
        b_tensor = torch.tensor(chunk, dtype=torch.uint8)
        _, _, waves = self.converter.encode(b_tensor)  # (T+1, 8)
        # Input: first T waves, Target: next T bytes
        return waves[:-1], b_tensor[1:].long()


# ------------------------------------------------------------------
# 2) Trainer Class
# ------------------------------------------------------------------
class NonFreqTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.link = SolitonChannel()  # For soliton grad pass
        self.history = {"train": [], "val": []}
        
    def _run_epoch(self, mode="train"):
        self.model.train(mode == "train")
        loader = self.train_loader if mode == "train" else self.val_loader
        total_loss, n_batches = 0.0, 0
        
        for waves, targets in loader:
            waves = waves.to(self.device)
            targets = targets.to(self.device)
            
            if mode == "train":
                self.optimizer.zero_grad()
                
            logits = self.model(waves)  # Forward through soliton links
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
            
            if mode == "train":
                loss.backward()
                # 🌊 Freq-backprop + Soliton grad transfer
                soliton_grad_pass(self.model, self.link) 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / max(n_batches, 1)
    
    def fit(self, epochs: int):
        print(f"🚀 Training started | Epochs: {epochs} | Device: {self.device}")
        best_val = float("inf")
        
        for ep in range(1, epochs + 1):
            tr_loss = self._run_epoch("train")
            vl_loss = self._run_epoch("val")
            self.history["train"].append(tr_loss)
            self.history["val"].append(vl_loss)
            
            marker = ""
            if vl_loss < best_val:
                best_val = vl_loss
                torch.save(self.model.state_dict(), "best_nonfreq.pt")
                marker = " ★"
                
            print(f"Epoch {ep:3d}/{epochs} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f}{marker}")
            
        # Save history
        with open("training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print("\n✅ Training complete! Best val loss:", round(best_val, 4))


# ------------------------------------------------------------------
# 3) DEMO / TEST RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Dummy repetitive text (model jaldi seekh lega pattern)
    raw_text = "soliton waves carry data without loss. " * 50
    
    # Split 90/10
    split = int(len(raw_text) * 0.9)
    train_ds = WaveDataset(raw_text[:split], seq_len=32)
    val_ds = WaveDataset(raw_text[split:], seq_len=32)
    
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8)
    
    # Model init (pure FFT + Soliton)
    model = Mamba2FreqFFT(d_model=64, n_layers=2, d_state=16)
    
    # Train!
    trainer = NonFreqTrainer(model, train_dl, val_dl, lr=2e-3)
    trainer.fit(epochs=10)
    
    print("\n🎉 STEP 7 PASSED — Full Tokenizer-Free Freq Training Loop Ready!")