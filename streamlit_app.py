import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ======================
# INDIKATOR
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
# SUPPORT/RESISTANCE
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

# ======================
# STREAMLIT APP
# ======================
st.title("📈 Trading Signals (EMA + RSI + MACD + Bollinger + S/R)")
st.sidebar.header("Pengaturan Data")

symbols_default = ["BTC-USD", "XAUUSD=X"]
symbols = st.sidebar.multiselect("Pilih simbol (bisa >1)", symbols_default, default=symbols_default)

custom = st.sidebar.text_input("Tambahkan ticker manual (opsional, pisahkan koma)")
if custom:
    symbols.extend([s.strip() for s in custom.split(",")])

timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)
period    = st.sidebar.selectbox("Periode", ["7d","1mo","3mo","6mo","1y"], index=1)

st.sidebar.caption("Catatan: yfinance membatasi beberapa kombinasi period/interval.")

# Tombol Hitung
if "hitung" not in st.session_state:
    st.session_state.hitung = False

if st.button("🔍 Hitung Sinyal"):
    st.session_state.hitung = True

# Eksekusi
if st.session_state.hitung:
    tab1, tab2 = st.tabs(["📊 Chart & Detail", "📋 Ringkasan Multi-Pair"])
    summary = []

    for sym in symbols:
        try:
            df = yf.download(sym, period=period, interval=timeframe)

            # kalau data kosong, coba fallback
            if df.empty:
                st.warning(f"{sym} → data kosong dengan {period}/{timeframe}, coba fallback (1mo/1h)")
                df = yf.download(sym, period="1mo", interval="1h")

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

            # ekspor CSV otomatis
            df.tail(200).to_csv(f"{sym.replace('=','')}_signals.csv")

        except Exception as e:
            st.error(f"{sym} → Error: {e}")
            summary.append((sym, f"Error"))

    with tab2:
        st.write("Ringkasan Sinyal Terakhir")
        st.table(pd.DataFrame(summary, columns=["Symbol","Signal"]))
else:
    st.info("Klik tombol **Hitung Sinyal** untuk mulai analisis.")
