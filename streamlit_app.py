import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ======================
# FUNGSI INDIKATOR
# ======================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain).rolling(period).mean()
    loss = pd.Series(loss).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, period=20, std_dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return ma, upper, lower

# ======================
# SUPPORT / RESISTANCE
# ======================
def sr_levels_from_swings(df, window=5):
    highs = (df['High'] == df['High'].rolling(window, center=True).max())
    lows = (df['Low'] == df['Low'].rolling(window, center=True).min())
    swing_highs = df.loc[highs, "High"]
    swing_lows = df.loc[lows, "Low"]
    return swing_highs.dropna().values, swing_lows.dropna().values

# ======================
# SIGNAL GENERATOR
# ======================
def compute_signals(df):
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["BB_MA"], df["BB_Upper"], df["BB_Lower"] = bollinger(df["Close"])
    sup, res = sr_levels_from_swings(df)
    return df, (sup, res)

def generate_signal(row):
    if row["EMA5"] > row["EMA20"] and row["RSI"] < 70 and row["MACD"] > row["MACD_signal"]:
        return "BUY"
    elif row["EMA5"] < row["EMA20"] and row["RSI"] > 30 and row["MACD"] < row["MACD_signal"]:
        return "SELL"
    else:
        return "HOLD"

# ======================
# CHART
# ======================
def plot_chart(df, levels, title="Chart"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Close", color="black")
    ax.plot(df.index, df["EMA5"], label="EMA5", color="blue")
    ax.plot(df.index, df["EMA20"], label="EMA20", color="orange")
    ax.plot(df.index, df["BB_Upper"], linestyle="--", color="green", alpha=0.5)
    ax.plot(df.index, df["BB_Lower"], linestyle="--", color="red", alpha=0.5)
    sup, res = levels
    for s in sup[-3:]:
        ax.axhline(s, linestyle=":", color="red", alpha=0.7)
    for r in res[-3:]:
        ax.axhline(r, linestyle=":", color="green", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    return fig

# ============
