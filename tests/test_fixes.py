"""One test per fix in diffusion_mt.py (tiny random model, runs in seconds)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diffusion_mt as dm  # noqa: E402

SRC = {"<pad>": 0, "<unk>": 1, "the": 2, "dog": 3, "bites": 4, "man": 5}
TGT = {"<pad>": 0, "<unk>": 1, "कुत्ता": 2, "आदमी": 3, "को": 4, "काटता": 5, "है": 6}
ITOS = {i: w for w, i in TGT.items()}


def _model():
    torch.manual_seed(0)
    m = dm.DiffusionTranslator(len(SRC), len(TGT), dm.Config(d_model=32, n_heads=2, ff_dim=64, max_len=8, dropout=0.0))
    return m.eval()


def test_fix1_generation_runs_and_is_deterministic():
    m = _model()
    a = dm.translate(m, ["the dog bites the man"], SRC, ITOS, "cpu", steps=10)
    b = dm.translate(m, ["the dog bites the man"], SRC, ITOS, "cpu", steps=10)
    assert isinstance(a[0], str) and a == b


def test_fix2_word_order_changes_the_condition():
    m = _model()
    s1 = torch.tensor([dm.encode("dog bites man", SRC, 8)])
    s2 = torch.tensor([dm.encode("man bites dog", SRC, 8)])
    (h1, _), (h2, _) = m.encode_source(s1), m.encode_source(s2)
    # same bag of words, different order -> different encodings only with positional encoding
    assert not torch.allclose(h1[0, :3].mean(0), h2[0, :3].mean(0), atol=1e-4)


def test_fix3_padding_does_not_leak_into_cross_attention():
    m = _model()
    src = torch.tensor([dm.encode("the dog bites", SRC, 8)])
    memory, mask = m.encode_source(src)
    memory2 = memory.clone()
    memory2[mask] = 123.0                     # garbage at PAD positions must be ignored
    x = torch.randn(1, 8, 32)
    t = torch.tensor([0.5])
    assert torch.allclose(m(x, t, memory, mask), m(x, t, memory2, mask), atol=1e-5)


def test_fix4_embeddings_are_trained():
    m = _model()
    names = {n for n, _ in m.named_parameters()}
    assert {"src_embed.weight", "tgt_embed.weight"} <= names
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    before = m.tgt_embed.weight.detach().clone()
    src = torch.tensor([dm.encode("the dog bites the man", SRC, 8)])
    tgt = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0]])
    m.train()
    loss, _ = dm.diffusion_loss(m, src, tgt)
    loss.backward()
    opt.step()
    assert (m.tgt_embed.weight - before).abs().max() > 0
