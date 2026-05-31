
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

INPUT_PATH = Path("data/news/headlines_2015_2020.parquet")
OUTPUT_PATH = Path("data/news/daily_sentiment_2015_2020.parquet")
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LEN = 64


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} headlines ({df['headline'].nunique():,} unique)")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    print(f"Loading {MODEL_NAME} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    # Map model label order -> our fixed [positive, negative, neutral]
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = next(i for i, l in id2label.items() if "pos" in l)
    neg_idx = next(i for i, l in id2label.items() if "neg" in l)
    neu_idx = next(i for i, l in id2label.items() if "neu" in l)

    # Score only unique headlines, then map back (saves a lot of compute).
    unique = df["headline"].drop_duplicates().tolist()
    n = len(unique)
    probs = np.zeros((n, 3), dtype=np.float32)  # columns: pos, neg, neu

    start = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = unique[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN
            ).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs[i : i + len(batch), 0] = p[:, pos_idx]
            probs[i : i + len(batch), 1] = p[:, neg_idx]
            probs[i : i + len(batch), 2] = p[:, neu_idx]
            if (i // BATCH_SIZE) % 20 == 0:
                done = min(i + len(batch), n)
                rate = done / max(1e-6, time.time() - start)
                eta = (n - done) / max(1e-6, rate)
                print(f"  {done:,}/{n:,}  ({rate:.0f} hl/s, ETA {eta/60:.1f} min)", flush=True)

    score_map = pd.DataFrame(
        {"headline": unique, "p_pos": probs[:, 0], "p_neg": probs[:, 1], "p_neu": probs[:, 2]}
    )
    df = df.merge(score_map, on="headline", how="left")
    df["compound"] = df["p_pos"] - df["p_neg"]

    daily = (
        df.groupby(["symbol", "date"])
        .agg(
            news_compound=("compound", "mean"),
            news_pos=("p_pos", "mean"),
            news_neg=("p_neg", "mean"),
            news_neu=("p_neu", "mean"),
            news_count=("compound", "size"),
        )
        .reset_index()
    )
    daily["news_has"] = 1.0
    daily["news_count"] = daily["news_count"].astype(float)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT_PATH, index=False)

    print("\n================ SUMMARY ================")
    print(f"Scored unique headlines : {n:,}  in {(time.time()-start)/60:.1f} min")
    print(f"Daily sentiment rows    : {len(daily):,}")
    print(f"Saved to                : {OUTPUT_PATH}")
    print(f"\nmean compound by ticker (most positive/negative tilt):")
    print(daily.groupby("symbol")["news_compound"].mean().sort_values(ascending=False).to_string())
    print(f"\noverall mean compound: {daily['news_compound'].mean():.4f}")


if __name__ == "__main__":
    main()
