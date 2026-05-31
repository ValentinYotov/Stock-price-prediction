"""
Streamlit demo app for the stock-forecasting Transformer baseline.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_sequences  # noqa: E402
from src.data.feature_engineering import (  # noqa: E402
    add_forward_log_return,
    create_all_features,
)
from src.data.loader import load_raw_dataset  # noqa: E402
from src.data.news_sentiment import (  # noqa: E402
    load_daily_sentiment,
    merge_sentiment_features,
)
from src.data.preprocessor import preprocess_data  # noqa: E402
from src.models.transformer_model import StockTransformer  # noqa: E402
from src.simulation.engine import BacktestEngine  # noqa: E402
from src.simulation.metrics import compute_metrics  # noqa: E402
from src.simulation.rules import signal_from_return_band  # noqa: E402
from src.utils import config as _cfg  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Forecasting Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.5px; }
        [data-testid="stMetricValue"] { font-size: 32px; font-weight: 600; }
        [data-testid="stMetricLabel"] { font-size: 14px; color: #9aa0a6; }
        .small-note { color: #9aa0a6; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_config():
    config = load_config()
    checkpoint_name = getattr(config.paths, "checkpoint_file", "best_model_base.pt")
    checkpoint_path = _cfg.PROJECT_ROOT / config.paths.models_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the base model first (notebooks/01_train_model.ipynb)."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    input_dim = state_dict["input_projection.weight"].shape[1]
    d_model = state_dict["input_projection.weight"].shape[0]
    n_layers = len(
        [k for k in state_dict if "encoder.layers" in k and "self_attention.w_q.weight" in k]
    )
    d_ff = state_dict["encoder.layers.0.feed_forward.linear1.weight"].shape[0]

    model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=config.model.n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=config.model.dropout,
        activation=config.model.activation,
        prediction_horizon=config.data.prediction_horizon,
    )
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    epoch_raw = checkpoint.get("epoch")
    info = {
        "checkpoint": checkpoint_path.name,
        "mtime": time.ctime(checkpoint_path.stat().st_mtime),
        "epoch": (epoch_raw + 1) if isinstance(epoch_raw, int) else None,
        "val_loss": checkpoint.get("score"),
        "input_dim": input_dim,
        "d_model": d_model,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "n_params": n_params,
    }
    return model, config, info


@st.cache_data(show_spinner="Loading market data...")
def load_market_data():
    config = load_config()
    df = load_raw_dataset(config=config)
    df["date"] = pd.to_datetime(df["date"])
    if config.data.start_date:
        df = df[df["date"] >= pd.to_datetime(config.data.start_date)]
    return df


@st.cache_data(show_spinner="Loading simulation parameters...")
def load_sim_defaults():
    config_path = _cfg.PROJECT_ROOT / "configs" / "default_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    return raw_cfg.get("simulation", {})


# ---------------------------------------------------------------------------
# Backtest pipeline (mirrors notebook 13)
# ---------------------------------------------------------------------------
def build_features(
    ticker_df: pd.DataFrame, config, sentiment_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    df = ticker_df.copy()
    df, _ = preprocess_data(
        df,
        handle_missing=True,
        missing_method="forward_fill",
        handle_outliers_flag=True,
        outliers_method="clip",
        normalize=False,
        date_column="date",
        symbol_column="symbol",
    )
    df = create_all_features(
        df,
        price_column=config.data.price_column,
        windows=config.data.features.windows,
        lags=[1, 2, 3, 5, 10] if config.data.features.lag_features else [],
        add_technical=config.data.features.technical_indicators,
        add_lags=config.data.features.lag_features,
        add_temporal=config.data.features.temporal_features,
        add_volume=True,
        simplified=config.data.features.simplified,
        symbol_column="symbol",
    )
    if config.data.target_column == "log_return":
        df = add_forward_log_return(
            df,
            price_column=config.data.price_column,
            target_column=config.data.target_column,
            symbol_column="symbol",
        )
    if sentiment_df is not None:
        df = merge_sentiment_features(
            df, sentiment_df, date_column="date", symbol_column="symbol"
        )
    return df.dropna().sort_values("date").reset_index(drop=True)


def run_backtest_for_ticker(
    ticker: str,
    raw_df: pd.DataFrame,
    model: StockTransformer,
    config,
    initial_capital: float,
    position_size_pct: float,
    commission_pct: float,
    risk_free_rate_annual: float,
    sentiment_df: pd.DataFrame | None = None,
    start_date=None,
    end_date=None,
    leverage: float = 1.0,
):
    ticker_df = raw_df[raw_df["symbol"] == ticker]
    if len(ticker_df) < 200:
        raise ValueError(f"Too few rows for {ticker}: {len(ticker_df)}")

    full = build_features(ticker_df, config, sentiment_df=sentiment_df)
    if len(full) < config.data.context_length + 50:
        raise ValueError(f"Not enough rows after feature engineering: {len(full)}")

    price_col = config.data.price_column
    target_col = config.data.target_column
    ctx = config.data.context_length

    feature_columns = [
        c
        for c in full.columns
        if c not in {"date", "symbol", price_col, target_col}
        and full[c].dtype in ["float64", "int64", "float32", "int32"]
    ]
    expected_input = model.input_projection.in_features
    if len(feature_columns) != expected_input:
        raise ValueError(
            f"{ticker}: feature count {len(feature_columns)} != model input_dim {expected_input}"
        )

    n = len(full)
    train_end = int(n * config.data.train_split)
    test_start = int(n * (config.data.train_split + config.data.val_split))

    scaler = StandardScaler().fit(full.iloc[:train_end][feature_columns])
    scaled = full.copy()
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])

    seq_start = max(0, test_start - ctx + 1)
    seg = scaled.iloc[seq_start:].reset_index(drop=True)

    stacked = np.column_stack(
        [seg[feature_columns].values, seg[target_col].values.reshape(-1, 1)]
    )
    seq_X, seq_y = create_sequences(stacked, ctx, config.data.prediction_horizon)
    if len(seq_X) == 0:
        raise ValueError("No sequences could be built.")

    X = seq_X[:, :, :-1]
    actual_returns = (
        seq_y[:, -1, -1]
        if config.data.prediction_horizon == 1
        else seq_y[:, :, -1][:, -1]
    )
    prices = seg[price_col].values[ctx : ctx + len(seq_X)]
    dates = seg["date"].values[ctx : ctx + len(seq_X)]
    sentiment_series = (
        seg["news_compound"].values[ctx : ctx + len(seq_X)]
        if "news_compound" in seg.columns
        else None
    )

    pred_batches = []
    with torch.no_grad():
        for start in range(0, len(X), config.training.batch_size):
            batch_x = torch.FloatTensor(X[start : start + config.training.batch_size])
            pred_batches.append(model(batch_x).detach().cpu())
    predicted_returns = torch.cat(pred_batches).numpy().reshape(-1)

    if start_date is not None or end_date is not None:
        dates_dt = pd.to_datetime(dates)
        lo = pd.Timestamp(start_date) if start_date is not None else dates_dt.min()
        hi = pd.Timestamp(end_date) if end_date is not None else dates_dt.max()
        mask = np.asarray((dates_dt >= lo) & (dates_dt <= hi))
        if int(mask.sum()) < 80:
            raise ValueError(
                f"Selected window has only {int(mask.sum())} trading days; "
                "need at least 80 (volatility window + a few days to trade)."
            )
        prices = prices[mask]
        dates = dates[mask]
        predicted_returns = predicted_returns[mask]
        actual_returns = actual_returns[mask]
        if sentiment_series is not None:
            sentiment_series = sentiment_series[mask]

    engine = BacktestEngine(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        commission_pct=commission_pct,
    )
    result = engine.run_smart_long_only(
        prices=prices,
        predicted_returns=predicted_returns,
        dates=dates,
        leverage=leverage,
    )
    metrics = compute_metrics(
        result,
        initial_capital=initial_capital,
        risk_free_rate_annual=risk_free_rate_annual,
        prices=prices,
    )

    directional_acc = float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns)))
    buy_signals = sum(1 for t in result.trades if t.side == "buy")

    return {
        "ticker": ticker,
        "result": result,
        "metrics": metrics,
        "dates": dates,
        "prices": prices,
        "predicted_returns": predicted_returns,
        "actual_returns": actual_returns,
        "sentiment": sentiment_series,
        "entry_thr_curve": result.entry_thresholds,
        "exit_thr_curve": result.exit_thresholds,
        "buy_signal_pct": 100 * buy_signals / len(prices),
        "directional_acc_pct": 100 * directional_acc,
        "in_training_set": ticker in set(config.data.tickers),
        "leverage": leverage,
        "margin_call_count": result.margin_call_count,
        "account_wiped_at_date": result.account_wiped_at_date,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
COLOR_BH = "#f59e0b"        # buy & hold (orange, dotted)
COLOR_BASE = "#94a3b8"      # base model (light slate)
COLOR_NEWS = "#3b82f6"      # news model (blue)
COLOR_BUY = "#22c55e"
COLOR_SELL = "#ef4444"


def _drawdown(eq):
    peak = np.maximum.accumulate(eq)
    safe = np.where(peak <= 0, np.nan, peak)
    return np.nan_to_num(100 * (peak - eq) / safe, nan=0.0)


def plot_equity_and_drawdown(res_base, res_news):
    """Overlay B&H, base and news equity curves; matching drawdowns below."""
    dates = pd.to_datetime(res_base["dates"])
    prices = res_base["prices"]
    eq_base = res_base["result"].equity_curve
    eq_news = res_news["result"].equity_curve
    initial = eq_base[0] if len(eq_base) else 0
    bh = (initial / prices[0]) * prices if len(prices) else None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Portfolio value", "Drawdown %"),
    )
    if bh is not None:
        fig.add_trace(
            go.Scatter(x=dates, y=bh, name="Buy & hold",
                       line=dict(color=COLOR_BH, width=2, dash="dot")),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=dates, y=eq_base, name="Base (technical)",
                   line=dict(color=COLOR_BASE, width=2.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=eq_news, name="News (+ FinBERT sentiment)",
                   line=dict(color=COLOR_NEWS, width=2.5)),
        row=1, col=1,
    )

    if bh is not None:
        fig.add_trace(
            go.Scatter(x=dates, y=_drawdown(bh),
                       line=dict(color=COLOR_BH, width=1.2, dash="dot"),
                       showlegend=False),
            row=2, col=1,
        )
    fig.add_trace(
        go.Scatter(x=dates, y=_drawdown(eq_base),
                   line=dict(color=COLOR_BASE, width=1.5),
                   showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=_drawdown(eq_news),
                   line=dict(color=COLOR_NEWS, width=1.5),
                   showlegend=False),
        row=2, col=1,
    )

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1, autorange="reversed")
    return fig


def plot_predictions_dual(res_base, res_news):
    """Actual return + base prediction + news prediction (same timeline)."""
    dates = pd.to_datetime(res_base["dates"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=res_base["actual_returns"],
                   mode="lines", name="Actual log return",
                   line=dict(color="#64748b", width=1), opacity=0.55)
    )
    fig.add_trace(
        go.Scatter(x=dates, y=res_base["predicted_returns"],
                   mode="lines", name="Base prediction",
                   line=dict(color=COLOR_BASE, width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=res_news["predicted_returns"],
                   mode="lines", name="News prediction",
                   line=dict(color=COLOR_NEWS, width=2))
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        title="Next-day predictions vs actual log return",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def trades_dataframe(res) -> pd.DataFrame:
    dates = pd.to_datetime(res["dates"])
    rows = []
    for t in res["result"].trades:
        idx = t.date_idx
        rows.append(
            {
                "date": dates[idx] if idx < len(dates) else None,
                "side": t.side.upper(),
                "price": round(t.price, 4),
                "quantity": round(t.quantity, 4),
                "commission": round(t.commission, 2),
                "cash_after": round(t.cash_after, 2),
                "shares_after": round(t.shares_after, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# News experiment: the SAME main model (2015-2020) with vs without FinBERT news
# sentiment features. best_model_base.pt vs best_model_news.pt on identical data.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading news-enhanced model...")
def load_news_model():
    cfg = load_config()
    path = PROJECT_ROOT / cfg.paths.models_dir / "best_model_news.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    input_dim = sd["input_projection.weight"].shape[1]
    d_model = sd["input_projection.weight"].shape[0]
    n_layers = len(
        [k for k in sd if "encoder.layers" in k and "self_attention.w_q.weight" in k]
    )
    d_ff = sd["encoder.layers.0.feed_forward.linear1.weight"].shape[0]
    m = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=cfg.model.n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        prediction_horizon=cfg.data.prediction_horizon,
    )
    m.load_state_dict(sd)
    m.eval()
    return {"model": m, "val_loss": ckpt.get("score"), "input_dim": input_dim}


@st.cache_data(show_spinner="Loading daily sentiment...")
def load_sentiment_table():
    return load_daily_sentiment(config=load_config())


def plot_price_with_sentiment(res):
    """Price line with daily sentiment bars underneath (news variant)."""
    dates = pd.to_datetime(res["dates"])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.7, 0.3], subplot_titles=("Price", "Daily news sentiment"),
    )
    fig.add_trace(
        go.Scatter(x=dates, y=res["prices"], name="close",
                   line=dict(color="#e5e7eb", width=1.8)),
        row=1, col=1,
    )
    sent = res.get("sentiment")
    if sent is not None:
        colors = ["#22c55e" if s > 0 else "#ef4444" for s in sent]
        fig.add_trace(
            go.Bar(x=dates, y=sent, name="compound", marker_color=colors, opacity=0.6),
            row=2, col=1,
        )
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark", showlegend=False,
    )
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="compound", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
model, config, model_info = load_model_and_config()
raw_df = load_market_data()
sim_defaults = load_sim_defaults()
news_model = load_news_model()
sentiment_df = load_sentiment_table() if news_model is not None else None

available_tickers = sorted(raw_df["symbol"].unique())
news_syms = set(sentiment_df["symbol"].unique()) if sentiment_df is not None else set()


@st.cache_data(show_spinner=False)
def get_test_window_dates(ticker: str) -> np.ndarray:
    """Dates that fall inside the model's test window for a ticker.

    Built without news features so that the bounds do not depend on whether
    the news-enhanced model is used; news only affects the feature vector,
    not which days are testable.
    """
    df = raw_df[raw_df["symbol"] == ticker]
    if len(df) < 200:
        return np.array([], dtype="datetime64[ns]")
    full = build_features(df, config, sentiment_df=None)
    ctx = config.data.context_length
    if len(full) < ctx + 50:
        return np.array([], dtype="datetime64[ns]")
    n = len(full)
    test_start = int(n * (config.data.train_split + config.data.val_split))
    seq_start = max(0, test_start - ctx + 1)
    seg = full.iloc[seq_start:].reset_index(drop=True)
    n_seq = len(seg) - ctx - config.data.prediction_horizon + 1
    if n_seq < 1:
        return np.array([], dtype="datetime64[ns]")
    return seg["date"].values[ctx : ctx + n_seq]


with st.sidebar:
    st.title("⚙️  Controls")

    st.markdown("**Ticker**")
    default_idx = 0
    for preferred in ("NFLX", "AAPL", "MSFT"):
        if preferred in available_tickers:
            default_idx = available_tickers.index(preferred)
            break
    ticker = st.selectbox(
        "Symbol",
        options=available_tickers,
        index=default_idx,
        format_func=lambda t: f"{t}  📰" if t in news_syms else t,
        label_visibility="collapsed",
        help="📰 marks tickers that have news sentiment available.",
    )

    ticker_dates = get_test_window_dates(ticker) if ticker else np.array([])
    if len(ticker_dates) > 0:
        total_days = len(ticker_dates)
        period_options = [("Full window", total_days)]
        for n_days in (200, 150, 100):
            if n_days < total_days:
                period_options.append((f"Last {n_days} days", n_days))
        labels = [lbl for lbl, _ in period_options]
        chosen_label = st.selectbox(
            "Period (anchored to most recent test day)",
            options=labels,
            index=0,
        )
        chosen_n = dict(period_options)[chosen_label]
        start_date = pd.Timestamp(ticker_dates[-chosen_n]).date()
        end_date = pd.Timestamp(ticker_dates[-1]).date()
        st.caption(
            f"Available: {pd.Timestamp(ticker_dates[0]).date()} → "
            f"{pd.Timestamp(ticker_dates[-1]).date()} "
            f"({total_days} trading days). Using {chosen_n}."
        )
    else:
        start_date, end_date = None, None
        st.warning(f"Not enough data for {ticker}.")

    st.markdown("---")
    st.markdown("**Portfolio**")
    initial_capital = st.number_input(
        "Initial capital ($)",
        min_value=1000.0,
        max_value=10_000_000.0,
        value=float(sim_defaults.get("initial_capital", 100_000)),
        step=1000.0,
    )
    position_size_pct = st.slider(
        "Position size %",
        min_value=0.05,
        max_value=1.0,
        value=float(sim_defaults.get("position_size_pct", 0.3)),
        step=0.05,
        help="Fraction of available cash committed on each buy.",
    )
    commission_pct = st.slider(
        "Commission %",
        min_value=0.0,
        max_value=1.0,
        value=float(sim_defaults.get("commission_pct", 0.1)),
        step=0.05,
        help="Round-trip commission per trade (industry typical: 0.05% - 0.20%).",
    )
    leverage = st.slider(
        "Leverage (x)",
        min_value=1.0,
        max_value=50.0,
        value=1.0,
        step=0.5,
        help=(
            "Borrowing multiplier on each position. 1x = no leverage. "
            "Higher leverage amplifies both gains and losses on the position. "
            "If a position's equity drops to zero, only that position is "
            "margin-called (you lose the committed margin); the rest of the "
            "cash stays free for the next trade. Trading stops only when the "
            "entire account is wiped out."
        ),
    )
    if leverage > 1.0:
        margin_wipe_pct = 100.0 / leverage
        st.caption(
            f":zap: At {leverage:g}x leverage, a {margin_wipe_pct:.1f}% adverse move on a "
            f"position loses the committed margin ({position_size_pct*100:.0f}% of cash)."
        )

    st.markdown("---")
    run = st.button(
        "▶  Run backtest",
        use_container_width=True,
        type="primary",
        disabled=(ticker is None or start_date is None),
    )


risk_free = float(sim_defaults.get("risk_free_rate_annual", 0.03))


st.title("📈  Stock Forecasting — base vs news model")
st.markdown(
    "<div class='small-note'>The same Transformer trained on 2015-2020, compared with and without "
    "6 FinBERT daily-sentiment features. Each run backtests <b>both models on the same ticker and "
    "period</b> against Buy &amp; Hold.</div>",
    unsafe_allow_html=True,
)

with st.expander("How the trading strategy decides — click to expand", expanded=False):
    st.markdown(
        """
**Smart long-biased strategy** (fixed, opinionated — no parameter tuning by hand).

Every day the system:

1. **Smooths the model prediction** with a 3-day exponential moving average to suppress single-day noise without lagging the signal too much.
2. **Converts it to a risk-adjusted z-score** by dividing by the trailing 20-day realised volatility — the model's edge is measured *in units of risk*, not raw return.
3. **Calibrates entry/exit thresholds adaptively** from each model's *own* past z-score distribution (walk-forward, no look-ahead). The base and news models predict on different scales, so model-agnostic thresholds are essential for a fair comparison.
4. **Enters a long position** only when *all three* hold:
    - the z-score is **above the model's own median** (top 50% of its past signal),
    - realised volatility is not in a black-swan regime (current vol < 2.5× the median over the last 30 days),
    - at least **1 day has passed** since the last sell.
5. **Exits** on the *first* of:
    - **Trailing stop** — price drops 6% from the post-entry peak,
    - **Take profit** — price is up 12% from the entry price,
    - **Signal exit** — z-score drops below its bottom 20% (the model turns relatively bearish),
    - **Time stop** — held for 30 trading days.

This is intentionally **long-biased**: in bull markets the system gives up some upside to enforce risk control; in flat or bearish markets it tends to beat Buy &amp; Hold by sidestepping the worst legs. All calculations are **causal** — only past prices and predictions are used.
        """
    )

st.markdown("---")


if news_model is None:
    st.warning(
        "News-enhanced model not found. Build it first:\n\n"
        "1. `py -3.11 scripts/fetch_fnspid_news.py`\n"
        "2. `py -3.11 scripts/score_news_finbert.py`\n"
        "3. `py -3.11 scripts/train_news.py`  (creates `best_model_news.pt`)"
    )
else:
    if "last_pair" not in st.session_state:
        st.session_state["last_pair"] = None

    placeholder_status = st.empty()

    if run and ticker is not None:
        try:
            with st.spinner(f"Backtesting {ticker} (base + news)..."):
                res_base = run_backtest_for_ticker(
                    ticker=ticker, raw_df=raw_df, model=model, config=config,
                    initial_capital=initial_capital, position_size_pct=position_size_pct,
                    commission_pct=commission_pct, risk_free_rate_annual=risk_free,
                    sentiment_df=None,
                    start_date=start_date, end_date=end_date,
                    leverage=leverage,
                )
                res_news = run_backtest_for_ticker(
                    ticker=ticker, raw_df=raw_df, model=news_model["model"], config=config,
                    initial_capital=initial_capital, position_size_pct=position_size_pct,
                    commission_pct=commission_pct, risk_free_rate_annual=risk_free,
                    sentiment_df=sentiment_df,
                    start_date=start_date, end_date=end_date,
                    leverage=leverage,
                )
            st.session_state["last_pair"] = (res_base, res_news)
        except Exception as e:
            placeholder_status.error(f"Error for {ticker}: {e}")
            st.session_state["last_pair"] = None

    pair = st.session_state["last_pair"]
    if pair is None:
        st.info("Pick a ticker from the left panel and press **Run backtest**.")
    else:
        res_base, res_news = pair
        mb, mn = res_base["metrics"], res_news["metrics"]
        news_days = (
            int(np.sum(res_news["sentiment"] != 0)) if res_news["sentiment"] is not None else 0
        )
        window_start = pd.to_datetime(res_base["dates"]).min().date()
        window_end = pd.to_datetime(res_base["dates"]).max().date()
        news_tag = (
            f" — {news_days} days with news"
            if ticker in news_syms
            else " — no news for this ticker"
        )
        st.subheader(
            f"{res_base['ticker']}  ·  {window_start} → {window_end}  "
            f"({len(res_base['prices'])} trading days){news_tag}"
        )

        for label, res in [("Base", res_base), ("News", res_news)]:
            mc = int(res.get("margin_call_count", 0))
            wiped = res.get("account_wiped_at_date")
            if mc > 0:
                st.warning(
                    f":zap: **{label} model had {mc} margin call{'s' if mc != 1 else ''}** "
                    f"at {res['leverage']:g}x leverage (each one lost the position's margin, "
                    f"trading continued with remaining cash)."
                )
            if wiped is not None:
                wiped_date = pd.to_datetime(wiped).date()
                st.error(
                    f":skull: **{label} model account was fully wiped out on {wiped_date}** "
                    f"at {res['leverage']:g}x leverage. No more capital, no further trades."
                )

        bh = mb.buy_and_hold_return_pct
        c1, c2, c3 = st.columns(3)
        c1.markdown("#### 🟠 Buy & hold")
        c1.metric("Return", f"{bh:+.2f}%" if bh is not None else "-")
        c1.metric("Trades", "—")
        c1.metric("Sharpe (ann.)", "—")
        c1.metric("Directional acc.", "—")

        c2.markdown("#### ⚪ Base (technical)")
        c2.metric(
            "Return",
            f"{mb.total_return_pct:+.2f}%",
            delta=f"{mb.total_return_pct - (bh or 0):+.2f} pp vs B&H" if bh is not None else None,
        )
        c2.metric("Trades", f"{mb.num_trades}")
        c2.metric("Sharpe (ann.)", f"{mb.sharpe_ratio_annual:.3f}")
        c2.metric("Directional acc.", f"{res_base['directional_acc_pct']:.1f}%")

        c3.markdown("#### 🔵 News (+ sentiment)")
        c3.metric(
            "Return",
            f"{mn.total_return_pct:+.2f}%",
            delta=f"{mn.total_return_pct - mb.total_return_pct:+.2f} pp vs Base",
        )
        c3.metric("Trades", f"{mn.num_trades}", delta=f"{mn.num_trades - mb.num_trades:+d}")
        c3.metric(
            "Sharpe (ann.)",
            f"{mn.sharpe_ratio_annual:.3f}",
            delta=f"{mn.sharpe_ratio_annual - mb.sharpe_ratio_annual:+.3f}",
        )
        c3.metric(
            "Directional acc.",
            f"{res_news['directional_acc_pct']:.1f}%",
            delta=f"{res_news['directional_acc_pct'] - res_base['directional_acc_pct']:+.1f} pp",
        )

        st.plotly_chart(plot_equity_and_drawdown(res_base, res_news), use_container_width=True)
        st.plotly_chart(plot_predictions_dual(res_base, res_news), use_container_width=True)

        if news_days > 0:
            st.plotly_chart(plot_price_with_sentiment(res_news), use_container_width=True)

        with st.expander("Trade logs"):
            tab_base, tab_news = st.tabs(["Base", "News"])
            with tab_base:
                df_b = trades_dataframe(res_base)
                if df_b.empty:
                    st.write("No trades.")
                else:
                    st.dataframe(df_b, use_container_width=True, height=320)
            with tab_news:
                df_n = trades_dataframe(res_news)
                if df_n.empty:
                    st.write("No trades.")
                else:
                    st.dataframe(df_n, use_container_width=True, height=320)

        base_vl = (
            f"{model_info['val_loss']:.6f}" if model_info["val_loss"] is not None else "n/a"
        )
        news_vl = (
            f"{news_model['val_loss']:.6f}" if news_model["val_loss"] is not None else "n/a"
        )
        st.caption(
            f"Models: best_model_base.pt (val_loss={base_vl}, {model_info['input_dim']} feat) vs "
            f"best_model_news.pt (val_loss={news_vl}, {news_model['input_dim']} feat). "
            f"Context window: {config.data.context_length} days."
        )
