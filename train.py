"""Train the English -> Hindi text-diffusion translator.

    python train.py --data data/train_data1.json --epochs 8

Prints train / validation loss per epoch, runs a generation check on
validation sentences, and saves checkpoints/diffusion_mt.pt and
checkpoints/history.json.
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

import diffusion_mt as dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_data1.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training pairs")
    ap.add_argument("--gen-check", type=int, default=200, help="validation sentences to generate")
    ap.add_argument("--steps", type=int, default=50, help="reverse diffusion steps")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    tr_s, tr_t, va_s, va_t = dm.load_pairs(args.data)
    if args.limit:
        tr_s, tr_t = tr_s[: args.limit], tr_t[: args.limit]
    print(f"train pairs: {len(tr_s):,}  validation pairs: {len(va_s):,}")

    src_vocab = dm.build_vocab(tr_s, args.min_freq)
    tgt_vocab = dm.build_vocab(tr_t, args.min_freq)
    print(f"vocab  en: {len(src_vocab):,}  hi: {len(tgt_vocab):,}")

    train_dl = DataLoader(dm.ParallelDataset(tr_s, tr_t, src_vocab, tgt_vocab, args.max_len),
                          batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(dm.ParallelDataset(va_s, va_t, src_vocab, tgt_vocab, args.max_len),
                        batch_size=args.batch_size)

    cfg = dm.Config(len(src_vocab), len(tgt_vocab), max_len=args.max_len)
    model = dm.TextDiffusionTranslator(cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    # model.parameters() now includes both embedding tables
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    os.makedirs(args.out, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for src, tgt in train_dl:
            loss, _ = dm.diffusion_loss(model, src.to(device), tgt.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
        train_loss = run / len(train_dl)
        val_loss = dm.evaluate_loss(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  ({time.time() - t0:.0f}s)")
        dm.save_checkpoint(f"{args.out}/diffusion_mt.pt", model, src_vocab, tgt_vocab)

    first, last = history[0]["val_loss"], history[-1]["val_loss"]
    print(f"validation loss {first:.2f} -> {last:.2f} ({(first - last) / first:.1%} lower than epoch 1)")

    k = min(args.gen_check, len(va_s))
    failures, outs = dm.generation_check(model, va_s[:k], src_vocab, tgt_vocab, device, args.steps)
    print(f"generation check: {k - failures}/{k} sentences produced output, {failures} failures")
    for s, ref, o in list(zip(va_s, va_t, outs))[:5]:
        print(f"\nEN : {s}\nREF: {ref}\nOUT: {o}")

    with open(f"{args.out}/history.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history, "generation_failures": failures,
                   "generation_checked": k}, f, indent=2)


if __name__ == "__main__":
    main()
