# LLM-Based Language Translator: Text Diffusion for English → Hindi

A generative **text diffusion model** for English → Hindi neural machine translation. A
**Transformer denoiser** learns to remove Gaussian noise from Hindi sentence embeddings,
conditioned on the English sentence through **cross-attention**. Instead of writing the
translation one word at a time like an autoregressive model, it starts from pure noise and
refines the whole sentence in parallel over 50 denoising steps.

Trained on an **80,797-pair English–Hindi parallel corpus** (80K sentences).

## How it works

```
English tokens ──► embedding + positional encoding ──► Transformer encoder ─────────┐
                                                                                    │ cross-attention
Hindi tokens ──► embedding (x0) ──► add noise at level t ──► x_t ──► Transformer denoiser ──► x0_hat ──► nearest words
                                                       timestep embedding(t) ──┘
```

- **Forward (noising) process:** `x_t = sqrt(1 - t) · x0 + sqrt(t) · ε`, with `t ∈ (0, 1)` and `ε ~ N(0, I)`.
- **Denoiser:** 3 Transformer layers, each with self-attention over the noisy Hindi tokens and
  cross-attention to the 3-layer English encoder. It predicts the clean embeddings `x0`
  (the parameterisation used by Diffusion-LM, Li et al. 2022). The noise estimate is recovered from it.
- **Loss:** denoising MSE plus a rounding cross-entropy that keeps every embedding decodable back to its word.
- **Generation:** start from noise at `t = 1`, then step down to `t = 0` with a DDIM-style update,
  snapping the `x0` estimate to the nearest word embeddings at each step. Decoding stops at `<eos>`.

## Debugging story: from v1 to v2

The first version (`notebooks/diffusion_LLMs_v1_original.ipynb`) trained, but its translations
were random strings of Hindi words. Tracing it back found one inference-breaking bug and three
architectural gaps:

| # | Problem in v1 | Why it broke the model | Fix in v2 |
|---|---|---|---|
| 1 | **No positional encoding** | Attention is order-agnostic, so the model saw each sentence as a bag of words. That matters for Hindi, whose word order (SOV) differs from English (SVO). | Sinusoidal positional encodings on both languages |
| 2 | **No padding masks** | Sentences are padded to 64 tokens, so attention and the loss were dominated by `PAD`. | `key_padding_mask` on the English side; `PAD` positions down-weighted in the loss |
| 3 | **Untrained embeddings** | `en_embed` / `hi_embed` were created outside the model, and the optimizer only received `model.parameters()`. The embeddings stayed at their random initial values, so the model learned to denoise meaningless vectors. | Embedding tables are model parameters, so they are trained, with a rounding loss that stops them collapsing |
| 4 | **Inference-breaking bug** | The sampling loop ran `t` from 0.02 **up** to 1.0, the opposite of the reverse process. Decoding also referenced an undefined variable (`weights`), so generation crashed. | Reverse process runs `t = 1 → 0` through the predicted `x0`; decoding fixed |

`tests/test_model.py` locks each fix in place: embeddings receive gradients, word order changes the
encoding, padded positions cannot affect the output, the sampler runs from noise to clean, and a
small model trained on a toy corpus translates without generation failures.

## Repository

| Path | Contents |
|---|---|
| `diffusion_mt.py` | Data loading, vocabulary, model, diffusion loss, sampler, evaluation helpers |
| `train.py` | Command-line training: per-epoch train and validation loss, generation check, checkpoint |
| `diffusion_mt.ipynb` | v2 notebook: data → model → 8-epoch training → loss curve → generation check → BLEU / chrF++ |
| `notebooks/diffusion_LLMs_v1_original.ipynb` | Original v1 notebook, kept for the before/after comparison |
| `tests/test_model.py` | Regression tests for the four fixes (CPU, a few seconds) |
| `data/README.md` | Expected format of `train_data1.json` |

## Run

```bash
pip install -r requirements.txt
# put the corpus at data/train_data1.json (see data/README.md)
python train.py --data data/train_data1.json --epochs 8
pytest -q tests
```

Or open `diffusion_mt.ipynb` (Kaggle or Colab with a GPU recommended).

`train.py` prints training and validation loss for every epoch. At the end it translates validation
sentences and reports how many produced output (generation failures = crashes or empty outputs). It
saves `checkpoints/diffusion_mt.pt` and `checkpoints/history.json`.

## Limitations and next steps

- Word-level tokenisation. Subword tokenisation (SentencePiece / BPE) would cut unknown words.
- Loss measures learning, not translation quality. The next step is BLEU / chrF++ against an
  autoregressive baseline (e.g. a fine-tuned mBART or IndicTrans2). The notebook includes the evaluation cell.
- Fixed-length (64-token) generation; length is learned implicitly through `<eos>` and `<pad>`.

## References

- Li, X. L. et al. (2022). *Diffusion-LM Improves Controllable Text Generation.* NeurIPS.
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, J., Meng, C., Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
