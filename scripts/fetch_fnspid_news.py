"""
Step 1 of the 2015-2020 news pipeline.

Downloads the FNSPID `All_external.csv` (~5.7 GB, cached by huggingface_hub),
streams it in chunks, and keeps only the rows for the thesis ticker universe
within 2015-2020. We persist just `symbol, date, headline` so the heavy CSV can
be deleted afterwards. Sentiment scoring (FinBERT/VADER) happens in step 2.

Run:
    py -3.11 scripts/fetch_fnspid_news.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from src.utils.config import load_config

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_FILE = "Stock_news/All_external.csv"
START_YEAR = 2015
END_YEAR = 2020  # inclusive; main price data ends 2020-09-30
OUTPUT_PATH = Path("data/news/headlines_2015_2020.parquet")
CHUNK_ROWS = 500_000


def main() -> None:
    config = load_config()
    tickers = set(config.data.tickers)
    print(f"Tickers ({len(tickers)}): {sorted(tickers)}")

    from huggingface_hub import hf_hub_download

    print("Downloading FNSPID All_external.csv (cached after first run)...", flush=True)
    csv_path = hf_hub_download(
        repo_id=FNSPID_REPO,
        filename=FNSPID_FILE,
        repo_type="dataset",
    )
    print(f"CSV cached at: {csv_path}", flush=True)

    kept = []
    total_rows = 0
    reader = pd.read_csv(
        csv_path,
        usecols=["Date", "Article_title", "Stock_symbol"],
        chunksize=CHUNK_ROWS,
        dtype=str,
        on_bad_lines="skip",
        engine="c",
    )
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        chunk = chunk[chunk["Stock_symbol"].isin(tickers)]
        if chunk.empty:
            if i % 10 == 0:
                print(f"  chunk {i}: scanned {total_rows:,} rows, kept {sum(len(k) for k in kept):,}", flush=True)
            continue
        dt = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
        chunk = chunk.assign(_date=dt.dt.tz_localize(None).dt.normalize())
        chunk = chunk.dropna(subset=["_date"])
        chunk = chunk[(chunk["_date"].dt.year >= START_YEAR) & (chunk["_date"].dt.year <= END_YEAR)]
        if not chunk.empty:
            kept.append(
                chunk.rename(columns={"Stock_symbol": "symbol", "Article_title": "headline"})[
                    ["symbol", "_date", "headline"]
                ].rename(columns={"_date": "date"})
            )
        print(f"  chunk {i}: scanned {total_rows:,} rows, kept {sum(len(k) for k in kept):,}", flush=True)

    if not kept:
        print("No matching rows found.")
        return

    df = pd.concat(kept, ignore_index=True)
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].str.strip()
    df = df[df["headline"] != ""]
    df = df.drop_duplicates(subset=["symbol", "date", "headline"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("\n================ SUMMARY ================")
    print(f"Total scanned rows : {total_rows:,}")
    print(f"Kept headlines     : {len(df):,}")
    print(f"Unique headlines   : {df['headline'].nunique():,}")
    print(f"Date range         : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Saved to           : {OUTPUT_PATH}")
    print("\nHeadlines per ticker:")
    print(df["symbol"].value_counts().to_string())
    print("\nHeadlines per year:")
    print(df["date"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
