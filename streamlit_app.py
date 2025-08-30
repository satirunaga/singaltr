import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# =========================
# 🔧 INDICATORS
# =========================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def bollinger_bands(series, period=20, std_mult=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =========================
# 🔧 SUPPORT / RESISTANCE
# =========================
def find_swings(df, left=3, right=3):
    win = left + right + 1
    highs = (
        df["High"]
        .rolling(win, center=True)
        .apply(lambda x: 1.0 if np.argmax(x) == left else 0.0, raw=True)
        .fillna(0)
        .astype(bool)
    )
    lows = (
        df["Low"]
        .rolling(win, center=True)
        .apply(lambda x: 1.0 if np.argmin(x) == left else 0.0, raw=True)
        .fillna(0)
        .astype(bool)
    )
    highs = highs.reindex(df.index).fillna(False).astype(bool)
    lows = lows.reindex(df.index).fillna(False).astype(bool)
    return highs, lows

def cluster_levels(levels, tol=0.002):
    out = []
    for l in levels:
        if not out or all(abs(l - x) > tol for x in out):
            out.append(l)
    return out

def sr_levels_from_swings(df, left=3, right=3, tol_mult=0.5):
    if len(df) < left + right + 2:
        return [], []
    highs, lows = find_swings(df, left, right)
    swing_highs, swing_lows = df.loc[highs, "High"], df.loc[lows, "Low"]

    _atr = atr(df, 14).dropna()
    tol = (_atr.iloc[-1] if len(_atr) > 0 else (df["High"] - df["Low"]).tail(14).mean()) * tol_mult
    return cluster_levels(swing_lows, tol), cluster_levels(swing_highs, tol)

# =========================
# 🔧 SIGNALS
# =========================
def compute_signals(df):
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"] = macd(df["Close"])
    df["BB_mid"], df["BB_up"], df["BB_low"] = bollinger_bands(df["Close"])
    sup, res = sr_levels_from_swings(df)
    return df, {"support": sup, "resistance": res}

def generate_signal(row):
    if row["EMA5"] > row["EMA20"] and row["RSI"] < 70 and row["MACD"] > row["MACD_signal"]:
        return "BUY"
    elif row["EMA5"] < row["EMA20"] and row["RSI"] > 30 and row["MACD"] < row["MACD_signal"]:
        return "SELL"
    return "HOLD"

# =========================
# 🔧 VISUALIZATION
# =========================
def plot_chart(df, levels, title="Chart"):
    mc = mpf.make_marketcolors(up="g", down="r", inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc)

    addplots = [
        mpf.make_addplot(df["EMA5"], color="blue"),
        mpf.make_addplot(df["EMA20"], color="orange"),
        mpf.make_addplot(df["BB_up"], color="gray"),
        mpf.make_addplot(df["BB_mid"], color="black"),
        mpf.make_addplot(df["BB_low"], color="gray"),
    ]

    fig, ax = mpf.plot(df, type="candle", style=s, volume=True,
                       addplot=addplots, returnfig=True, figsize=(10,6))
    for lvl in levels["support"]:
        ax[0].axhline(lvl, color="green", linestyle="--", alpha=0.5)
    for lvl in levels["resistance"]:
        ax[0].axhline(lvl, color="red", linestyle="--", alpha=0.5)
    ax[0].set_title(title)
    return fig

# =========================
# 🔧 STREAMLIT APP
# =========================
st.title("📈 Trading Signals (EMA + RSI + MACD + Bollinger + S/R)")
st.sidebar.header("Pengaturan Data")

symbols_default = ["BTC-USD", "EURUSD=X", "XAUUSD=X"]
symbols = st.sidebar.multiselect("Pilih simbol (bisa >1)", symbols_default, default=symbols_default)
custom = st.sidebar.text_input("Tambahkan ticker manual (opsional, pisahkan koma)")
if custom:
    symbols.extend([s.strip() for s in custom.split(",")])

timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)
period    = st.sidebar.selectbox("Periode", ["7d","1mo","3mo","6mo","1y"], index=1)

st.sidebar.caption("Catatan: yfinance membatasi beberapa kombinasi period/interval.")

tab1, tab2 = st.tabs(["📊 Chart & Detail", "📋 Ringkasan Multi-Pair"])
summary = []

for sym in symbols:
    try:
        df = yf.download(sym, period=period, interval=timeframe)
        if df.empty:
            summary.append((sym, "No data"))
            continue
        df, levels = compute_signals(df)
        df["Signal"] = df.apply(generate_signal, axis=1)
        last = df.iloc[-1]
        summary.append((sym, last["Signal"]))

        with tab1:
            st.subheader(sym)
            st.write(f"📌 Sinyal Terakhir: **{last['Signal']}**")
            fig = plot_chart(df, levels, title=sym)
            st.pyplot(fig)
            with st.expander("Dataframe"):
                st.dataframe(df.tail(20))

        df.tail(100).to_csv(f"{sym.replace('=','')}_signals.csv")

    except Exception as e:
        summary.append((sym, f"Error: {e}"))

with tab2:
    st.write("Ringkasan Sinyal Terakhir")
    st.table(pd.DataFrame(summary, columns=["Symbol","Signal"]))
