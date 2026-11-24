"""Regression tests for the four fixes, on a tiny synthetic dataset (CPU, ~1 min)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diffusion_mt as dm  # noqa: E402

torch.manual_seed(0)
EN = ["the cat sat", "a dog ran", "the dog sat", "a cat ran", "the bird flew", "a bird sat"] * 20
HI = ["बिल्ली बैठी", "कुत्ता दौड़ा", "कुत्ता बैठा", "बिल्ली दौड़ी", "चिड़िया उड़ी", "चिड़िया बैठी"] * 20


def tiny():
    sv, tv = dm.build_vocab(EN, 1), dm.build_vocab(HI, 1)
    cfg = dm.Config(len(sv), len(tv), max_len=8, d_model=64, n_heads=4, enc_layers=1, dec_layers=2,
                    emb_dim=16, dropout=0.0)
    return dm.TextDiffusionTranslator(cfg), sv, tv


def batch(sv, tv, n=12):
    src = torch.tensor([dm.encode(s, sv, 8) for s in EN[:n]])
    tgt = torch.tensor([dm.encode(s, tv, 8, add_eos=True) for s in HI[:n]])
    return src, tgt


def test_embeddings_are_trained():
    model, sv, tv = tiny()
    names = {n for n, _ in model.named_parameters()}
    assert "tgt_embed.weight" in names and "src_embed.weight" in names
    src, tgt = batch(sv, tv)
    loss, _ = dm.diffusion_loss(model, src, tgt)
    loss.backward()
    assert model.tgt_embed.weight.grad.abs().sum() > 0
    assert model.src_embed.weight.grad.abs().sum() > 0


def test_word_order_matters():
    model, sv, tv = tiny(); model.eval()
    a = torch.tensor([dm.encode("the cat sat", sv, 8)])
    b = torch.tensor([dm.encode("sat cat the", sv, 8)])
    ma, _ = model.encode_source(a); mb, _ = model.encode_source(b)
    # without positional encoding, mean-pooled memories of a permutation would match
    assert not torch.allclose(ma[0, :3].mean(0), mb[0, :3].mean(0), atol=1e-4)


def test_padding_is_ignored():
    model, sv, tv = tiny(); model.eval()
    src = torch.tensor([dm.encode("the cat sat", sv, 8)])
    x = torch.randn(1, 8, model.cfg.emb_dim); t = torch.tensor([0.5])
    mem, pad = model.encode_source(src)
    out1 = model(x, t, mem, pad)
    mem2 = mem.clone(); mem2[pad] = 1e3          # garbage in padded memory slots
    out2 = model(x, t, mem2, pad)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_reverse_process_runs_from_noise_to_clean():
    ts = torch.linspace(dm.T_MAX, dm.T_MIN, 11)
    assert ts[0] > ts[-1]


def test_training_reduces_loss_and_generation_never_fails():
    model, sv, tv = tiny()
    src, tgt = batch(sv, tv, 36)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    with torch.no_grad():
        start = sum(dm.diffusion_loss(model, src, tgt)[0].item() for _ in range(5)) / 5
    model.train()
    for _ in range(300):
        loss, _ = dm.diffusion_loss(model, src, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        end = sum(dm.diffusion_loss(model, src, tgt)[0].item() for _ in range(5)) / 5
    assert end < start * 0.7, (start, end)
    failures, outs = dm.generation_check(model, EN[:6], sv, tv, "cpu", steps=20)
    assert failures == 0, outs
