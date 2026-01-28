from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import load_dataset

from src.utils.config import Config, load_config


def load_raw_dataset(
    config: Optional[Config] = None, split: Optional[str] = None
) -> pd.DataFrame:
    if config is None:
        config = load_config()
    dataset_name = config.data.dataset_name
    
    local_file = Path("data/raw/sp500_stocks_data.parquet")
    
    if local_file.exists():
        print(f"Зареждане на локален dataset от: {local_file}")
        df = pd.read_parquet(local_file)
        print(f"Заредено! Размер: {df.shape}")
    else:
        print(f"Зареждане на dataset от Hugging Face: {dataset_name}")
        print("Това може да отнеме няколко минути при първо зареждане...")
        
        try:
            # Опитваме се да заредим само price данните (без news)
            # Използваме data_files за да укажем конкретния файл
            print("Опитвам се да заредя само price данните...")
            try:
                dataset = load_dataset(
                    dataset_name,
                    data_files="sp500_daily_ratios_20yrs.zip",
                    download_mode="reuse_cache_if_exists"
                )
                print("Успешно зареден price dataset!")
            except Exception as e1:
                print(f"Грешка при зареждане с data_files: {e1}")
                print("Опитвам се да заредя целия dataset и да филтрирам...")
                # Fallback: опитваме се да заредим целия dataset
                try:
                    dataset = load_dataset(dataset_name, download_mode="reuse_cache_if_exists")
                    # Ако dataset-ът е dict с multiple splits, вземаме първия
                    if isinstance(dataset, dict):
                        # Вземаме split който има price данни (обикновено 'train')
                        split_name = list(dataset.keys())[0]
                        dataset = dataset[split_name]
                except Exception as e3:
                    raise e1  # Повдигаме оригиналната грешка
        except Exception as e:
            print(f"Грешка при зареждане: {e}")
            print("Опитвам се алтернативен метод...")
            try:
                # Последен опит - директно от zip файла
                from huggingface_hub import hf_hub_download
                import zipfile
                import io
                
                print("Сваляне на zip файла директно...")
                zip_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename="sp500_daily_ratios_20yrs.zip",
                    repo_type="dataset"
                )
                
                print(f"Zip файл свален. Четене на CSV...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        first_csv = csv_files[0]
                        with zip_ref.open(first_csv) as f:
                            df = pd.read_csv(f)
                            print(f"CSV зареден успешно! Размер: {df.shape}")
                            # Запазваме локално за следващи пъти
                            local_file.parent.mkdir(parents=True, exist_ok=True)
                            df.to_parquet(local_file, index=False)
                            print(f"Запазено локално в: {local_file}")
                            return df
            except Exception as e2:
                print(f"Грешка: {e2}")
                print("\nПроблем: Не може да се свърже с Hugging Face Hub.")
                print(f"Моля, свали dataset-а ръчно в notebook-а и запази го в: {local_file}")
                raise
        
        if split is None:
            split_name = list(dataset.keys())[0]
        else:
            split_name = split
        
        print(f"Dataset зареден. Конвертиране в pandas...")
        df = dataset[split_name].to_pandas()
        print(f"Конвертирано! Размер: {df.shape}")
    
    column_mapping = {
        "Ticker": "symbol",
        "Date": "date",
        "Open": "open",
        "Close": "close",
        "Volume": "volume",
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    if "high" not in df.columns and "open" in df.columns and "close" in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    if "low" not in df.columns and "open" in df.columns and "close" in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)
    
    return df


def filter_by_tickers_and_dates(
    df: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol_column: str = "symbol",
    date_column: str = "date",
) -> pd.DataFrame:
    result = df
    if tickers is not None:
        result = result[result[symbol_column].isin(list(tickers))]
    if start_date is not None or end_date is not None:
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result = result.copy()
            result[date_column] = pd.to_datetime(result[date_column])
        if start_date is not None:
            result = result[result[date_column] >= pd.to_datetime(start_date)]
        if end_date is not None:
            result = result[result[date_column] <= pd.to_datetime(end_date)]
    return result


def load_and_filter_dataset(
    config: Optional[Config] = None,
    tickers: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    split: Optional[str] = None,
) -> pd.DataFrame:
    if config is None:
        config = load_config()
    if tickers is None:
        tickers = config.data.tickers
    if start_date is None:
        start_date = config.data.start_date
    if end_date is None:
        end_date = config.data.end_date
    df = load_raw_dataset(config=config, split=split)
    if "Ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"Ticker": "symbol"})
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    
    print(f"Филтриране на данни... Първоначален размер: {df.shape}")
    df = filter_by_tickers_and_dates(
        df,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        symbol_column="symbol",
        date_column="date",
    )
    print(f"Филтрирано! Финален размер: {df.shape}")
    return df


__all__ = [
    "load_raw_dataset",
    "filter_by_tickers_and_dates",
    "load_and_filter_dataset",
]

