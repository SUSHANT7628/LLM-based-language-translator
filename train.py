"""Train the fixed diffusion translator and record validation loss per epoch.

    python train.py --data train_data1.json --epochs 8          # writes results/
    python train.py --data train_data1.json --epochs 1 --limit 2000   # quick check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import diffusion_mt as dm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CS779 JSON file (e.g. train_data1.json)")
    ap.add_argument("--pair", default="English-Hindi")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training pairs (0 = all)")
    ap.add_argument("--n-generate", type=int, default=200, help="validation sentences to translate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs, n_file = dm.load_pairs(args.data, args.pair, seed=args.seed)
    if args.limit:
        train_pairs = train_pairs[:args.limit]
    src_vocab = dm.build_vocab([s for s, _ in train_pairs], args.min_freq)
    tgt_vocab = dm.build_vocab([t for _, t in train_pairs], args.min_freq)
    tgt_itos = {i: w for w, i in tgt_vocab.items()}
    cfg = dm.Config(max_len=args.max_len)
    model = dm.DiffusionTranslator(len(src_vocab), len(tgt_vocab), cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.src_embed.weight.numel() + model.tgt_embed.weight.numel()
    print(f"device={device} pairs in file={n_file:,} train={len(train_pairs):,} val={len(val_pairs):,} "
          f"src_vocab={len(src_vocab):,} tgt_vocab={len(tgt_vocab):,} params={n_params/1e6:.1f}M")

    train_dl = DataLoader(dm.PairDataset(train_pairs, src_vocab, tgt_vocab, cfg.max_len),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dm.PairDataset(val_pairs, src_vocab, tgt_vocab, cfg.max_len), batch_size=128)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    history = []
    before = dm.evaluate(model, val_dl, device)
    history.append({"epoch": 0, **before})
    print(f"epoch 0  val_loss={before['val_loss']:.4f}  (mse={before['val_mse']:.3f} round_ce={before['val_round_ce']:.3f})")
    emb_before = model.tgt_embed.weight.detach().clone()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0.0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            loss, _ = dm.diffusion_loss(model, src, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * src.size(0)
            n += src.size(0)
        ev = dm.evaluate(model, val_dl, device)
        history.append({"epoch": ep, "train_loss": tot / n, **ev, "elapsed_min": round((time.time() - t0) / 60, 1)})
        print(f"epoch {ep}  train_loss={tot/n:.4f}  val_loss={ev['val_loss']:.4f}  "
              f"(mse={ev['val_mse']:.3f} round_ce={ev['val_round_ce']:.3f})  ({(time.time()-t0)/60:.1f} min)", flush=True)
        dm.save_checkpoint(out / "checkpoint.pt", model, src_vocab, tgt_vocab, history)

    # generation on held-out sentences
    sample = val_pairs[:args.n_generate]
    failures, hyps = 0, []
    for s, _ in sample:
        try:
            h = dm.translate(model, [s], src_vocab, tgt_itos, device)[0]
        except Exception as e:  # noqa: BLE001 - count any crash as a failure
            h = f"<error: {type(e).__name__}>"
        failures += dm.is_failure(h) or h.startswith("<error")
        hyps.append(h)

    metrics = {
        "pair": args.pair,
        "pairs_in_file": n_file,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "src_vocab": len(src_vocab), "tgt_vocab": len(tgt_vocab),
        "parameters": n_params, "trainable_parameters": trained, "embedding_parameters": emb,
        "embeddings_updated": bool((model.tgt_embed.weight.detach().cpu() - emb_before.cpu()).abs().max() > 0),
        "config": vars(args) | {"model": dm.asdict(cfg)},
        "device": str(device),
        "history": history,
        "val_loss_first_epoch": history[1]["val_loss"],
        "val_loss_last_epoch": history[-1]["val_loss"],
        "val_loss_reduction_pct": 100 * (1 - history[-1]["val_loss"] / history[1]["val_loss"]),
        "generation": {"sentences": len(sample), "failures": int(failures)},
    }
    try:
        import sacrebleu
        refs = [t for _, t in sample]
        metrics["generation"]["bleu"] = sacrebleu.corpus_bleu(hyps, [refs]).score
        metrics["generation"]["chrf"] = sacrebleu.corpus_chrf(hyps, [refs]).score
    except ImportError:
        pass
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(out / "sample_translations.tsv", "w", encoding="utf-8") as fh:
        fh.write("english\treference_hindi\tmodel_hindi\n")
        for (s, t), h in zip(sample, hyps):
            fh.write(f"{s}\t{t}\t{h}\n")
    print(json.dumps({k: metrics[k] for k in ("val_loss_first_epoch", "val_loss_last_epoch",
                                              "val_loss_reduction_pct", "generation")}, indent=2))


if __name__ == "__main__":
    main()
