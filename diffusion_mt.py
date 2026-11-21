"""Text-diffusion neural machine translation (English -> Hindi).

A Transformer denoiser learns to remove Gaussian noise from the embeddings of a Hindi sentence,
conditioned on the English sentence through cross-attention. Generation starts from pure noise,
denoises step by step, and maps each vector to the nearest Hindi word.

This is the fixed version of original/diffusion_LLMs_original.ipynb. The original had one
inference-breaking bug and three architectural gaps, each marked FIX below:

  FIX 1  generate() referenced an undefined variable (`weights` instead of `vocab_weights`), so
         inference crashed. Its reverse loop also ran t upward (0 -> 1) with x <- x - eps/steps,
         which is not a valid reverse diffusion step. Replaced by a deterministic DDIM sampler
         that runs t from 1 to 0.
  FIX 2  No positional encoding: self-attention is permutation-invariant, so word order was
         invisible. Sinusoidal positions are added to both the source and the noisy target.
  FIX 3  No padding masks: attention attended to PAD tokens in the English condition. The source
         key-padding mask is now passed to the encoder and to every cross-attention layer.
  FIX 4  Untrained embeddings: the embedding tables were created outside the model, so the
         optimizer (Adam over model.parameters()) never updated them. They now live inside the
         module, with rounding and anchor losses (Li et al., Diffusion-LM, 2022) so the trainable
         target embeddings stay distinct instead of collapsing to make the denoising loss trivial.

Following Diffusion-LM, the denoiser predicts the clean embeddings x0 rather than the noise: text
has to be rounded back to discrete words, and that step needs x0.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PAD, UNK = 0, 1


# ----------------------------------------------------------------------------- data
def load_pairs(path: str, pair: str = "English-Hindi", val_fraction: float = 0.05, seed: int = 0):
    """Read the CS779 JSON ({pair: {split: {id: {source, target}}}}).

    Uses the file's Validation split if it has targets; otherwise holds out `val_fraction` of the
    training pairs with a fixed seed.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    train, val = [], []
    for split, entries in data[pair].items():
        for e in entries.values():
            if "source" not in e or "target" not in e:
                continue
            (val if split.lower().startswith("val") else train).append((e["source"], e["target"]))
    n_file_train = len(train)
    if not val:
        rng = random.Random(seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        n_val = int(round(val_fraction * len(train)))
        val = [train[i] for i in idx[:n_val]]
        train = [train[i] for i in idx[n_val:]]
    return train, val, n_file_train


def tokenize(sentence: str) -> list[str]:
    return sentence.lower().split()


def build_vocab(sentences, min_freq: int = 2) -> dict[str, int]:
    counter = Counter(w for s in sentences for w in tokenize(s))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(sentence: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(w, UNK) for w in tokenize(sentence)][:max_len]
    return ids + [PAD] * (max_len - len(ids))


class PairDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len):
        self.src = torch.tensor([encode(s, src_vocab, max_len) for s, _ in pairs])
        self.tgt = torch.tensor([encode(t, tgt_vocab, max_len) for _, t in pairs])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


# ----------------------------------------------------------------------------- model
@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 1
    dec_layers: int = 3
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 32
    t_eps: float = 1e-3
    round_weight: float = 1.0
    round_tokens: int = 512        # positions per batch used for the rounding loss


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


def timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    """t in [0, 1] -> (B, d) sinusoidal features."""
    half = d // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = 1000.0 * t[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class DiffusionTranslator(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # FIX 4: embeddings are module parameters, so the optimizer trains them
        self.src_embed = nn.Embedding(src_vocab_size, d, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d)
        # FIX 2: positional encoding
        self.register_buffer("pos", sinusoidal(cfg.max_len, d), persistent=False)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True),
            cfg.enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        # Transformer denoiser: self-attention over the noisy Hindi sequence + cross-attention to English
        self.denoiser = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True),
            cfg.dec_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.round_scale = nn.Parameter(torch.tensor(10.0))

    # x0 = target word embeddings, scaled so each dimension has unit variance like the noise
    def tgt_table(self) -> torch.Tensor:
        return F.normalize(self.tgt_embed.weight, dim=-1) * math.sqrt(self.cfg.d_model)

    def embed_target(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        return self.tgt_table()[tgt_ids]

    def encode_source(self, src_ids: torch.Tensor):
        src_mask = src_ids == PAD                                    # FIX 3
        L = src_ids.size(1)
        h = self.src_embed(src_ids) + self.pos[:L]          # N(0, 1) init: no sqrt(d) scaling
        return self.encoder(h, src_key_padding_mask=src_mask), src_mask

    def forward(self, x_t, t, memory, src_mask):
        """Predict the clean target embeddings x0 from x_t at diffusion time t, given the English sentence."""
        L = x_t.size(1)
        h = x_t + self.pos[:L] + self.time_mlp(timestep_embedding(t, self.cfg.d_model))[:, None]
        h = self.denoiser(h, memory, memory_key_padding_mask=src_mask)   # FIX 3 in cross-attention
        return self.out(h)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Score vectors against every Hindi word embedding (cosine similarity x learned scale)."""
        return self.round_scale * F.normalize(x, dim=-1) @ F.normalize(self.tgt_embed.weight, dim=-1).T


def noise(x0, t, eps):
    a = (1.0 - t).sqrt()[:, None, None]
    s = t.sqrt()[:, None, None]
    return a * x0 + s * eps


def diffusion_loss(model: DiffusionTranslator, src, tgt, t=None, eps=None, generator=None):
    """Loss on one batch. The denoiser predicts the clean embeddings x0 (Diffusion-LM style).

    loss = MSE(x0_hat, x0)                      denoising
         + CE(logits(x0_hat), target words)     rounding: x0_hat must decode to the right words
         + CE(logits(x0), target words)         anchor: keeps the trainable embeddings distinct
    Target-side PAD positions are kept: the model must learn where the sentence ends.
    Returns (total, parts) where parts holds each term as a float.
    """
    cfg = model.cfg
    B = src.size(0)
    x0 = model.embed_target(tgt)
    if t is None:
        t = torch.rand(B, device=src.device, generator=generator) * (1 - 2 * cfg.t_eps) + cfg.t_eps
    if eps is None:
        eps = torch.randn(x0.shape, device=src.device, generator=generator)
    x_t = noise(x0, t, eps)
    memory, src_mask = model.encode_source(src)
    x0_hat = model(x_t, t, memory, src_mask)
    mse = F.mse_loss(x0_hat, x0)
    flat_tgt = tgt.reshape(-1)
    k = min(cfg.round_tokens, flat_tgt.numel())
    pick = torch.randperm(flat_tgt.numel(), device=src.device, generator=generator)[:k]
    d = x0.size(-1)
    round_ce = F.cross_entropy(model.logits(x0_hat.reshape(-1, d)[pick]), flat_tgt[pick])
    anchor_ce = F.cross_entropy(model.logits(x0.reshape(-1, d)[pick]), flat_tgt[pick])
    total = mse + cfg.round_weight * round_ce + anchor_ce
    return total, {"mse": mse.item(), "round_ce": round_ce.item(), "anchor_ce": anchor_ce.item()}


@torch.no_grad()
def evaluate(model, loader, device, n_t: int = 8, seed: int = 1234) -> dict:
    """Deterministic validation loss: fixed grid of diffusion times, fixed noise, fixed token picks."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sums = {"val_loss": 0.0, "val_mse": 0.0, "val_round_ce": 0.0}
    n = 0
    ts = torch.linspace(0.05, 0.95, n_t)
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        for t_val in ts:
            t = torch.full((src.size(0),), float(t_val), device=device)
            eps = torch.randn(src.size(0), tgt.size(1), model.cfg.d_model, generator=g).to(device)
            total, parts = diffusion_loss(model, src, tgt, t=t, eps=eps, generator=g)
            sums["val_loss"] += total.item() * src.size(0)
            sums["val_mse"] += parts["mse"] * src.size(0)
            sums["val_round_ce"] += parts["round_ce"] * src.size(0)
            n += src.size(0)
    return {k: v / n for k, v in sums.items()}


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def translate(model, sentences, src_vocab, tgt_itos, device, steps: int = 50, seed: int = 0,
              batch_size: int = 64) -> list[str]:
    """FIX 1: deterministic DDIM sampling from t = 1 down to t = 0.

    At each step the denoiser predicts the clean embeddings x0_hat; the implied noise is
    (x_t - sqrt(1-t) x0_hat) / sqrt(t), and x moves to the next (smaller) t. The final x0_hat is
    rounded to the nearest Hindi word at every position, and PAD positions are dropped.
    """
    model.eval()
    cfg = model.cfg
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b:b + batch_size]
        src = torch.tensor([encode(s, src_vocab, cfg.max_len) for s in chunk], device=device)
        memory, src_mask = model.encode_source(src)
        x = torch.randn(len(chunk), cfg.max_len, cfg.d_model, generator=g).to(device)
        ts = torch.linspace(1 - cfg.t_eps, cfg.t_eps, steps + 1, device=device)
        for k in range(steps):
            t, t_next = ts[k], ts[k + 1]
            x0_hat = model(x, t.expand(len(chunk)), memory, src_mask)
            eps_dir = (x - (1 - t).sqrt() * x0_hat) / t.sqrt()
            x = (1 - t_next).sqrt() * x0_hat + t_next.sqrt() * eps_dir
        ids = model.logits(x0_hat).argmax(-1).tolist()
        out.extend(" ".join(tgt_itos[i] for i in row if i != PAD) for row in ids)
    return out


def is_failure(text: str) -> bool:
    """A generation failure: empty output, or nothing but unknown-word tokens."""
    words = text.split()
    return len(words) == 0 or all(w == "<unk>" for w in words)


def save_checkpoint(path, model, src_vocab, tgt_vocab, history):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "history": history}, path)
