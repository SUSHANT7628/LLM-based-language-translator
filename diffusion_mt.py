"""Continuous (embedding-space) text diffusion for English -> Hindi translation.

A Transformer denoiser learns to remove Gaussian noise from Hindi sentence
embeddings, conditioned on the English sentence through cross-attention.

Fixes relative to the first version (notebooks/diffusion_LLMs_v1_original.ipynb):
  1. Positional encoding  - sinusoidal positions added to both languages;
                            attention alone is order-agnostic.
  2. Padding masks        - PAD tokens are masked out of attention
                            (key_padding_mask) and down-weighted in the loss.
  3. Trained embeddings   - the embedding tables live inside the model, so the
                            optimizer updates them (they were frozen at random
                            init before), with a rounding loss that keeps them
                            decodable and prevents collapse.
  4. Inference bug        - the reverse process now runs t = 1 -> 0 (it ran
                            0 -> 1 before), uses a DDIM-style update through the
                            predicted x0, and the decoding step no longer
                            references an undefined variable.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
PAD_ID, UNK_ID, EOS_ID = 0, 1, 2


# --------------------------------------------------------------------------- data
def load_pairs(path: str, pair: str = "English-Hindi"):
    """Read train_data1.json -> (train_src, train_tgt, val_src, val_tgt)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tr_s, tr_t, va_s, va_t = [], [], [], []
    for split, entries in data[pair].items():
        for entry in entries.values():
            if split == "Validation":
                va_s.append(entry["source"]); va_t.append(entry["target"])
            else:
                tr_s.append(entry["source"]); tr_t.append(entry["target"])
    return tr_s, tr_t, va_s, va_t


def split_tokens(sentence: str):
    return sentence.lower().split()


def build_vocab(sentences, min_freq: int = 2, max_size: int | None = None):
    counter = Counter(tok for s in sentences for tok in split_tokens(s))
    vocab = {PAD: PAD_ID, UNK: UNK_ID, EOS: EOS_ID}
    for tok, freq in counter.most_common(max_size):
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(sentence: str, vocab: dict, max_len: int, add_eos: bool = False):
    ids = [vocab.get(t, UNK_ID) for t in split_tokens(sentence)]
    ids = ids[: max_len - 1] + [EOS_ID] if add_eos else ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


class ParallelDataset(Dataset):
    def __init__(self, src, tgt, src_vocab, tgt_vocab, max_len: int):
        self.src, self.tgt = src, tgt
        self.src_vocab, self.tgt_vocab, self.max_len = src_vocab, tgt_vocab, max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        s = encode(self.src[i], self.src_vocab, self.max_len)
        t = encode(self.tgt[i], self.tgt_vocab, self.max_len, add_eos=True)
        return torch.tensor(s), torch.tensor(t)


# -------------------------------------------------------------------------- model
@dataclass
class Config:
    src_vocab_size: int
    tgt_vocab_size: int
    max_len: int = 64
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    dropout: float = 0.1
    emb_dim: int = 128          # diffusion runs in this (smaller) embedding space
    pad_loss_weight: float = 0.1
    rounding_weight: float = 1.0


def sinusoidal(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t in [0, 1], shape (B,) -> (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TextDiffusionTranslator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # FIX 3: embeddings are module parameters -> trained by the optimizer
        self.src_embed = nn.Embedding(cfg.src_vocab_size, d, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.emb_dim)
        nn.init.normal_(self.tgt_embed.weight, std=1.0)  # same scale as the noise
        # FIX 1: positional encodings
        self.register_buffer("pos", sinusoidal(cfg.max_len, d), persistent=False)

        self.time_mlp = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.in_proj = nn.Linear(cfg.emb_dim, d)
        self.out_proj = nn.Linear(d, cfg.emb_dim)

        enc = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.enc_layers)
        # denoiser: self-attention over noisy Hindi + cross-attention to English
        dec = nn.TransformerDecoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerDecoder(dec, cfg.dec_layers)
        self.norm = nn.LayerNorm(d)

    # x0 for the diffusion process: normalised target embeddings
    def target_x0(self, tgt_ids):
        return self.tgt_embed(tgt_ids)

    def encode_source(self, src_ids):
        src_pad = src_ids.eq(PAD_ID)                       # FIX 2: source padding mask
        h = self.src_embed(src_ids) * math.sqrt(self.cfg.d_model) + self.pos[: src_ids.size(1)]
        return self.encoder(h, src_key_padding_mask=src_pad), src_pad

    def forward(self, x_t, t, memory, src_pad):
        """Denoise: predict the clean embeddings x0 from x_t. x_t: (B, L, emb_dim), t: (B,).

        x0-parameterisation (as in Diffusion-LM, Li et al. 2022) trains far more
        reliably for text than predicting the noise; the noise estimate is
        recovered from it when sampling."""
        h = self.in_proj(x_t) + self.pos[: x_t.size(1)]
        h = h + self.time_mlp(timestep_embedding(t, self.cfg.d_model))[:, None, :]
        h = self.denoiser(h, memory, memory_key_padding_mask=src_pad)
        return self.out_proj(self.norm(h))

    def logits(self, x0_hat):
        """Rounding: map continuous vectors back to the Hindi vocabulary."""
        return x0_hat @ self.tgt_embed.weight.t()


# ---------------------------------------------------------------------- diffusion
T_MIN, T_MAX = 1e-3, 1.0 - 1e-3


def add_noise(x0, t, noise=None):
    """Forward process q(x_t | x0): x_t = sqrt(1-t) x0 + sqrt(t) eps."""
    noise = torch.randn_like(x0) if noise is None else noise
    tt = t.view(-1, 1, 1)
    return (1 - tt).sqrt() * x0 + tt.sqrt() * noise, noise


def eps_from_x0(x_t, x0_hat, t):
    tt = t.view(-1, 1, 1)
    return (x_t - (1 - tt).sqrt() * x0_hat) / tt.sqrt()


def diffusion_loss(model: TextDiffusionTranslator, src_ids, tgt_ids):
    """Denoising MSE on x0 (PAD positions down-weighted) + rounding cross-entropy."""
    cfg = model.cfg
    memory, src_pad = model.encode_source(src_ids)
    x0 = model.target_x0(tgt_ids)
    t = torch.rand(x0.size(0), device=x0.device) * (T_MAX - T_MIN) + T_MIN
    x_t, _ = add_noise(x0, t)
    x0_hat = model(x_t, t, memory, src_pad)

    w = torch.where(tgt_ids.eq(PAD_ID), cfg.pad_loss_weight, 1.0)          # FIX 2: PAD-aware loss
    mse = (((x0_hat - x0) ** 2).mean(-1) * w).sum() / w.sum()
    # rounding: predicted and clean embeddings must both decode to the right word,
    # which keeps the (now trainable) embedding table spread out instead of collapsing
    ce = F.cross_entropy(model.logits(x0_hat).transpose(1, 2), tgt_ids, reduction="none")
    ce0 = F.cross_entropy(model.logits(x0).transpose(1, 2), tgt_ids, reduction="none")
    ce = ((ce + ce0) * w).sum() / w.sum()
    return mse + cfg.rounding_weight * ce, {"mse": mse.item(), "ce": ce.item()}


@torch.no_grad()
def sample(model: TextDiffusionTranslator, src_ids, steps: int = 50, clamp: bool = True):
    """Reverse process, t = 1 -> 0 (FIX 4). DDIM-style deterministic update."""
    model.eval()
    memory, src_pad = model.encode_source(src_ids)
    B, L = src_ids.size(0), model.cfg.max_len
    x = torch.randn(B, L, model.cfg.emb_dim, device=src_ids.device)
    ts = torch.linspace(T_MAX, T_MIN, steps + 1, device=src_ids.device)
    for i in range(steps):
        t, t_next = ts[i].expand(B), ts[i + 1].expand(B)
        x0_hat = model(x, t, memory, src_pad)
        if clamp:  # snap the x0 estimate to the nearest word embeddings
            x0_hat = model.tgt_embed(model.logits(x0_hat).argmax(-1))
        eps_hat = eps_from_x0(x, x0_hat, t)
        x = (1 - t_next.view(-1, 1, 1)).sqrt() * x0_hat + t_next.view(-1, 1, 1).sqrt() * eps_hat
    return model.logits(x0_hat).argmax(-1)          # (B, L) token ids


def decode(ids, inv_vocab):
    out = []
    for i in ids:
        if i in (EOS_ID, PAD_ID):
            break
        out.append(inv_vocab.get(i, UNK))
    return " ".join(out)


def translate(model, sentences, src_vocab, tgt_vocab, steps: int = 50, device="cpu"):
    inv = {i: w for w, i in tgt_vocab.items()}
    src = torch.tensor([encode(s, src_vocab, model.cfg.max_len) for s in sentences], device=device)
    return [decode(row.tolist(), inv) for row in sample(model, src, steps)]


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for src, tgt in loader:
        loss, _ = diffusion_loss(model, src.to(device), tgt.to(device))
        total += loss.item() * src.size(0); n += src.size(0)
    return total / max(n, 1)


def generation_check(model, sentences, src_vocab, tgt_vocab, device, steps=50, batch=32):
    """Count generation failures: exceptions or empty outputs."""
    failures, outputs = 0, []
    for i in range(0, len(sentences), batch):
        chunk = sentences[i:i + batch]
        try:
            outs = translate(model, chunk, src_vocab, tgt_vocab, steps, device)
        except Exception:  # noqa: BLE001 - any crash is a failure
            failures += len(chunk); outputs += [""] * len(chunk); continue
        failures += sum(1 for o in outs if not o.strip())
        outputs += outs
    return failures, outputs


def save_checkpoint(path, model, src_vocab, tgt_vocab):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = TextDiffusionTranslator(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck["src_vocab"], ck["tgt_vocab"]
