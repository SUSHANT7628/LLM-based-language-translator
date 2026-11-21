# Text-Diffusion Neural Machine Translation (English → Hindi)

A generative **text diffusion model** for English → Hindi translation. A **Transformer denoiser**
learns to recover the embeddings of a Hindi sentence from Gaussian noise. It is conditioned on the
English sentence through **cross-attention**. To translate, the model starts from pure noise,
denoises step by step, and rounds every position to the nearest Hindi word.

Data: the English–Hindi part of the IIT Kanpur CS779 machine-translation corpus, which has
**80,797 parallel sentence pairs** (`train_data1.json`, not redistributed here).

## From a broken prototype to a working model

The first version is kept in [`original/diffusion_LLMs_original.ipynb`](original/diffusion_LLMs_original.ipynb).
It trained, but it could not translate: generation crashed, and the saved output for
"We are alive." is a string of unrelated Hindi words. I found one inference-breaking bug and three
architectural gaps. All four are fixed in [`diffusion_mt.py`](diffusion_mt.py), and each has a test
in [`tests/test_fixes.py`](tests/test_fixes.py).

| # | Problem in the original | Effect | Fix |
| --- | --- | --- | --- |
| 1 | **Inference-breaking bug.** `generate_hindi` used an undefined variable (`weights` instead of `vocab_weights`), and its reverse loop ran *t* upward with `x = x - pred_noise / steps` | Generation crashed. Even with the name fixed, the loop was not a reverse diffusion process | Deterministic DDIM sampler from *t* = 1 down to 0 |
| 2 | **No positional encoding** | Self-attention is permutation-invariant, so "dog bites man" = "man bites dog" | Sinusoidal positions added to the source and to the noisy target |
| 3 | **No padding masks** | Attention attended to the PAD tokens in the English condition | Source key-padding mask in the encoder and in every cross-attention layer |
| 4 | **Untrained embeddings** | The embedding tables were created outside the model, so `Adam(model.parameters())` never updated them. They stayed random | Embeddings moved inside the module. Rounding and anchor losses (Diffusion-LM) stop them collapsing |

## Model

```
English ids -> embedding + positions -> Transformer encoder (1 layer) -> memory
noisy Hindi embeddings x_t + positions + time embedding
   -> Transformer denoiser: 3 x [self-attention -> cross-attention to memory -> MLP]
   -> predicted clean embeddings x0_hat -> nearest Hindi word at each position
```

- Noise schedule: x_t = √(1−t) · x0 + √t · ε, with t ~ U(0, 1).
- Training loss: MSE(x0_hat, x0) + rounding cross-entropy (x0_hat must decode to the right words) + anchor cross-entropy (keeps word embeddings distinct).
- The denoiser predicts x0 rather than the noise, following Diffusion-LM (Li et al., 2022), because text has to be rounded back to discrete words.
- Sizes: word-level vocabularies (min. frequency 2), d_model 256, 4 heads, max length 32 words.

## Run

```bash
pip install -r requirements.txt
python train.py --data train_data1.json --epochs 8      # writes results/
python -m pytest -q                                      # the four fix tests
```

`results/metrics.json` records the validation loss after every epoch, the number of generation
failures on held-out sentences, and BLEU/chrF. `results/sample_translations.tsv` lists
English / reference / model output side by side. The validation loss uses a fixed grid of
diffusion times and fixed noise, so epochs are directly comparable.

## Results

_Pending: the 8-epoch training run on the CS779 corpus._

## Limitations

- Word-level vocabularies leave many Hindi word forms as unknown. Subword tokenisation (SentencePiece/BPE) is the obvious next step.
- A small diffusion model trained for a few epochs on one corpus is a proof of concept. It is not competitive with autoregressive Transformers on translation quality.
- Validation loss measures denoising, not translation quality. BLEU/chrF are reported alongside it.
