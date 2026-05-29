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
@st.cache_resource(show_spinner="Зареждане на модел...")
def load_model_and_config():
    config = load_config()
    checkpoint_name = getattr(config.paths, "checkpoint_file", "best_model_base.pt")
    checkpoint_path = _cfg.PROJECT_ROOT / config.paths.models_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the base model first (notebooks/03_train_model.ipynb)."
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


@st.cache_data(show_spinner="Зареждане на пазарни данни...")
def load_market_data():
    config = load_config()
    df = load_raw_dataset(config=config)
    df["date"] = pd.to_datetime(df["date"])
    if config.data.start_date:
        df = df[df["date"] >= pd.to_datetime(config.data.start_date)]
    return df


@st.cache_data(show_spinner="Зареждане на simulation параметри...")
def load_sim_defaults():
    config_path = _cfg.PROJECT_ROOT / "configs" / "default_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    return raw_cfg.get("simulation", {})


# ---------------------------------------------------------------------------
# Backtest pipeline (mirrors notebook 13)
# ---------------------------------------------------------------------------
def build_features(ticker_df: pd.DataFrame, config) -> pd.DataFrame:
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
    return df.dropna().sort_values("date").reset_index(drop=True)


def run_backtest_for_ticker(
    ticker: str,
    raw_df: pd.DataFrame,
    model: StockTransformer,
    config,
    entry_quantile: float,
    exit_quantile: float,
    initial_capital: float,
    position_size_pct: float,
    commission_pct: float,
    risk_free_rate_annual: float,
):
    ticker_df = raw_df[raw_df["symbol"] == ticker]
    if len(ticker_df) < 200:
        raise ValueError(f"Too few rows for {ticker}: {len(ticker_df)}")

    full = build_features(ticker_df, config)
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

    pred_batches = []
    with torch.no_grad():
        for start in range(0, len(X), config.training.batch_size):
            batch_x = torch.FloatTensor(X[start : start + config.training.batch_size])
            pred_batches.append(model(batch_x).detach().cpu())
    predicted_returns = torch.cat(pred_batches).numpy().reshape(-1)

    entry_thr = float(np.quantile(predicted_returns, entry_quantile))
    exit_thr = float(np.quantile(predicted_returns, exit_quantile))

    engine = BacktestEngine(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        commission_pct=commission_pct,
    )
    result = engine.run_from_log_returns(
        prices=prices,
        predicted_returns=predicted_returns,
        dates=dates,
        entry_threshold=entry_thr,
        exit_threshold=exit_thr,
        signal_mode="band",
    )
    metrics = compute_metrics(
        result,
        initial_capital=initial_capital,
        risk_free_rate_annual=risk_free_rate_annual,
        prices=prices,
    )

    buy_signals = sum(
        1
        for p in predicted_returns
        if signal_from_return_band(p, False, entry_thr, exit_thr) == "buy"
    )
    directional_acc = float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns)))

    return {
        "ticker": ticker,
        "result": result,
        "metrics": metrics,
        "dates": dates,
        "prices": prices,
        "predicted_returns": predicted_returns,
        "actual_returns": actual_returns,
        "entry_thr": entry_thr,
        "exit_thr": exit_thr,
        "buy_signal_pct": 100 * buy_signals / len(prices),
        "directional_acc_pct": 100 * directional_acc,
        "in_training_set": ticker in set(config.data.tickers),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
COLOR_STRATEGY = "#3b82f6"
COLOR_BH = "#f59e0b"
COLOR_BUY = "#22c55e"
COLOR_SELL = "#ef4444"


def plot_equity_and_drawdown(res):
    dates = pd.to_datetime(res["dates"])
    prices = res["prices"]
    equity = res["result"].equity_curve

    bh_shares = res["metrics"].buy_and_hold_return_pct is not None
    initial_capital = equity[0] if len(equity) else 0
    bh_equity = (initial_capital / prices[0]) * prices if bh_shares else None

    def dd(eq):
        peak = np.maximum.accumulate(eq)
        safe = np.where(peak <= 0, np.nan, peak)
        return np.nan_to_num(100 * (peak - eq) / safe, nan=0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Portfolio value", "Drawdown %"),
    )
    fig.add_trace(
        go.Scatter(x=dates, y=equity, name="Strategy", line=dict(color=COLOR_STRATEGY, width=2.5)),
        row=1,
        col=1,
    )
    if bh_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bh_equity,
                name="Buy & hold",
                line=dict(color=COLOR_BH, width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd(equity),
            name="Strategy DD",
            line=dict(color=COLOR_STRATEGY, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.15)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if bh_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd(bh_equity),
                name="B&H DD",
                line=dict(color=COLOR_BH, width=1.5, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1, autorange="reversed")
    return fig


def plot_predictions_with_trades(res):
    dates = pd.to_datetime(res["dates"])
    pred = res["predicted_returns"]
    actual = res["actual_returns"]
    trades = res["result"].trades

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            mode="lines",
            name="Actual log return",
            line=dict(color="#94a3b8", width=1),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred,
            mode="lines",
            name="Predicted log return",
            line=dict(color=COLOR_STRATEGY, width=2),
        )
    )
    fig.add_hline(y=res["entry_thr"], line=dict(color=COLOR_BUY, dash="dot"),
                  annotation_text=f"entry {res['entry_thr']:+.4f}")
    fig.add_hline(y=res["exit_thr"], line=dict(color=COLOR_SELL, dash="dot"),
                  annotation_text=f"exit {res['exit_thr']:+.4f}")

    if trades:
        buys = [(t.date_idx, t.price) for t in trades if t.side == "buy"]
        sells = [(t.date_idx, t.price) for t in trades if t.side == "sell"]
        if buys:
            idxs, _ = zip(*buys)
            fig.add_trace(
                go.Scatter(
                    x=dates[list(idxs)],
                    y=pred[list(idxs)],
                    mode="markers",
                    name="Buy",
                    marker=dict(color=COLOR_BUY, size=10, symbol="triangle-up"),
                )
            )
        if sells:
            idxs, _ = zip(*sells)
            fig.add_trace(
                go.Scatter(
                    x=dates[list(idxs)],
                    y=pred[list(idxs)],
                    mode="markers",
                    name="Sell",
                    marker=dict(color=COLOR_SELL, size=10, symbol="triangle-down"),
                )
            )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Predictions vs actual log return (test period)",
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
# UI
# ---------------------------------------------------------------------------
model, config, model_info = load_model_and_config()
raw_df = load_market_data()
sim_defaults = load_sim_defaults()

available_tickers = sorted(raw_df["symbol"].unique())
training_tickers = set(config.data.tickers)
training_list = sorted([t for t in available_tickers if t in training_tickers])
unseen_list = sorted([t for t in available_tickers if t not in training_tickers])


with st.sidebar:
    st.title("⚙️  Контроли")

    st.markdown("**Тикер**")
    group = st.radio(
        "Източник",
        options=["Training", "Unseen", "Всички"],
        index=1,
        horizontal=True,
    )
    if group == "Training":
        ticker_options = training_list
    elif group == "Unseen":
        ticker_options = unseen_list
    else:
        ticker_options = available_tickers
    default_idx = ticker_options.index("AAPL") if "AAPL" in ticker_options else 0
    ticker = st.selectbox("Тикер", options=ticker_options, index=default_idx)

    st.markdown("---")
    st.markdown("**Стратегия (band thresholds)**")
    entry_quantile = st.slider(
        "Entry quantile", min_value=0.50, max_value=0.95, value=0.70, step=0.05
    )
    exit_quantile = st.slider(
        "Exit quantile", min_value=0.05, max_value=0.50, value=0.30, step=0.05
    )

    st.markdown("---")
    st.markdown("**Портфолио**")
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
    )
    commission_pct = st.slider(
        "Commission %",
        min_value=0.0,
        max_value=1.0,
        value=float(sim_defaults.get("commission_pct", 0.1)),
        step=0.05,
    )

    st.markdown("---")
    run = st.button("▶  Run backtest", use_container_width=True, type="primary")


# Header
st.title("📈  Stock Forecasting — Base model")
st.markdown(
    "<div class='small-note'>Transformer baseline върху технически индикатори. "
    "Прогнозира next-day log return; band rule превръща прогнозите в trading сигнали.</div>",
    unsafe_allow_html=True,
)

# Model card
with st.container():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Checkpoint", model_info["checkpoint"])
    c2.metric(
        "Best val loss",
        f"{model_info['val_loss']:.6f}" if model_info["val_loss"] is not None else "-",
    )
    c3.metric("Architecture", f"d={model_info['d_model']} · L={model_info['n_layers']}")
    c4.metric("Parameters", f"{model_info['n_params']:,}")
    c5.metric("Training tickers", f"{len(training_tickers)}")

st.markdown("---")


# Results section
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

placeholder_status = st.empty()

if run:
    risk_free = float(sim_defaults.get("risk_free_rate_annual", 0.03))
    try:
        with st.spinner(f"Backtesting {ticker}..."):
            res = run_backtest_for_ticker(
                ticker=ticker,
                raw_df=raw_df,
                model=model,
                config=config,
                entry_quantile=entry_quantile,
                exit_quantile=exit_quantile,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                commission_pct=commission_pct,
                risk_free_rate_annual=risk_free,
            )
        st.session_state["last_result"] = res
    except Exception as e:
        placeholder_status.error(f"Грешка за {ticker}: {e}")
        st.session_state["last_result"] = None

res = st.session_state["last_result"]
if res is None:
    st.info("Избери тикер от левия панел и натисни **Run backtest**.")
else:
    m = res["metrics"]

    origin = "training" if res["in_training_set"] else "unseen"
    st.subheader(f"{res['ticker']} ({origin})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy return", f"{m.total_return_pct:+.2f}%")
    c2.metric(
        "Buy & hold",
        f"{m.buy_and_hold_return_pct:+.2f}%"
        if m.buy_and_hold_return_pct is not None
        else "-",
    )
    c3.metric(
        "Excess vs B&H",
        f"{m.excess_return_vs_bh_pct:+.2f}%"
        if m.excess_return_vs_bh_pct is not None
        else "-",
        delta=f"{m.excess_return_vs_bh_pct:+.2f}%"
        if m.excess_return_vs_bh_pct is not None
        else None,
    )
    c4.metric("Sharpe (ann.)", f"{m.sharpe_ratio_annual:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max drawdown", f"{m.max_drawdown_pct:.2f}%")
    c6.metric("Trades", f"{m.num_trades}")
    c7.metric("Buy signal freq.", f"{res['buy_signal_pct']:.1f}%")
    c8.metric("Directional acc.", f"{res['directional_acc_pct']:.1f}%")

    st.plotly_chart(plot_equity_and_drawdown(res), use_container_width=True)
    st.plotly_chart(plot_predictions_with_trades(res), use_container_width=True)

    with st.expander("Сделки (trade log)"):
        df_trades = trades_dataframe(res)
        if df_trades.empty:
            st.write("Няма сделки за този период.")
        else:
            st.dataframe(df_trades, use_container_width=True, height=320)

    val_loss_str = (
        f"{model_info['val_loss']:.6f}"
        if model_info["val_loss"] is not None
        else "n/a"
    )
    st.caption(
        f"Test window: {len(res['prices'])} дни. "
        f"Entry threshold: {res['entry_thr']:+.5f}, exit threshold: {res['exit_thr']:+.5f}. "
        f"Model: {model_info['checkpoint']} (val_loss={val_loss_str})."
    )
