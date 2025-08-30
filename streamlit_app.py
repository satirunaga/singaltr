# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Trading Signals: EMA + S/R + S/D", layout="wide")

# ----------------------------
# 0) Utility: data & mapping
# ----------------------------
TICKER_MAP = {
    # Forex
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X", "XAUUSD": "XAUUSD=X", "XAGUSD": "XAGUSD=X",
    # Crypto
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "LTCUSD": "LTC-USD",
    # Stocks (contoh)
    "AAPL": "AAPL", "TSLA": "TSLA", "MSFT": "MSFT"
}
DEFAULT_SYMBOLS = ["BTCUSD", "EURUSD", "XAUUSD"]

INTERVALS = ["1m","5m","15m","1h","1d"]
PERIODS   = ["1d","5d","1mo","3mo","6mo","1y"]

def map_ticker(sym: str) -> str:
    return TICKER_MAP.get(sym.upper(), sym)

@st.cache_data(show_spinner=False)
def fetch_data(yahoo_ticker: str, period="7d", interval="1h") -> pd.DataFrame:
    df = yf.download(yahoo_ticker, period=period, interval=interval, progress=False)
    if df.empty: 
        return df
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

# ----------------------------
# 1) Indicators
# ----------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ = series.ewm(span=fast, adjust=False).mean()
    slow_ = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ - slow_
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist

def bollinger(series, length=20, mult=2):
    mb = series.rolling(length).mean()
    sd = series.rolling(length).std()
    ub = mb + mult*sd
    lb = mb - mult*sd
    return mb, ub, lb

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ----------------------------
# 2) Swing High/Low & S/R
# ----------------------------
def find_swings(df: pd.DataFrame, left=3, right=3):
    """
    Menandai pivot high/low: center candle harus tertinggi/terendah di jendela (left+right+1).
    """
    win = left + right + 1
    highs = df["High"].rolling(win, center=True)\
            .apply(lambda x: 1.0 if np.argmax(x)==left else 0.0, raw=True).fillna(0).astype(bool)
    lows  = df["Low"].rolling(win, center=True)\
            .apply(lambda x: 1.0 if np.argmin(x)==left else 0.0, raw=True).fillna(0).astype(bool)
    return highs, lows

def cluster_levels(prices: pd.Series, tol: float):
    """
    Menggabungkan level yang berdekatan (dalam toleransi 'tol').
    Menghasilkan daftar level unik.
    """
    if len(prices)==0:
        return []
    levels = sorted(prices.values)
    clustered = []
    for p in levels:
        if not clustered:
            clustered.append(p)
        else:
            if abs(p - clustered[-1]) <= tol:
                # rata-rata untuk menggabungkan
                clustered[-1] = (clustered[-1] + p) / 2.0
            else:
                clustered.append(p)
    return clustered

def sr_levels_from_swings(df, left=3, right=3, tol_mult=0.5):
    """
    Buat level Support/Resistance dari pivot, dikelompokkan dengan toleransi berbasis ATR.
    """
    highs, lows = find_swings(df, left=left, right=right)
    swing_highs = df.loc[highs, "High"]
    swing_lows  = df.loc[lows,  "Low"]
    _atr = atr(df, 14).dropna()
    tol = (_atr.iloc[-1] if len(_atr)>0 else (df["High"]-df["Low"]).tail(14).mean()) * tol_mult
    res_levels = cluster_levels(swing_highs, tol)
    sup_levels = cluster_levels(swing_lows,  tol)
    return sup_levels, res_levels

# ----------------------------
# 3) Supply/Demand Zones (proxy)
# ----------------------------
def sd_zones_from_swings(df, atr_mult=1.0, max_zones=4):
    """
    Zona sederhana:
    - Demand: di sekitar swing low -> [low, low + ATR*mult]
    - Supply: di sekitar swing high -> [high - ATR*mult, high]
    Ambil beberapa zona terakhir saja.
    """
    highs, lows = find_swings(df, left=3, right=3)
    swings_hi = df.loc[highs, ["High"]].tail(max_zones)
    swings_lo = df.loc[lows,  ["Low"] ].tail(max_zones)
    _atr = atr(df, 14)
    last_atr = _atr.iloc[-1] if _atr.notna().any() else (df["High"]-df["Low"]).tail(14).mean()

    supply = []
    for idx, row in swings_hi.iterrows():
        hi = float(row["High"])
        supply.append((idx, hi - atr_mult*last_atr, hi + 1e-9))  # (time, low, high)

    demand = []
    for idx, row in swings_lo.iterrows():
        lo = float(row["Low"])
        demand.append((idx, lo - 1e-9, lo + atr_mult*last_atr))  # (time, low, high)

    return supply, demand

def price_in_zone(price, zone_low, zone_high):
    return (price >= zone_low) and (price <= zone_high)

# ----------------------------
# 4) Voting & Signal
# ----------------------------
def compute_signals(df: pd.DataFrame):
    close = df["Close"]
    df["EMA5"]  = ema(close, 5)
    df["EMA20"] = ema(close, 20)
    df["RSI14"] = rsi(close, 14)
    df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = macd(close)
    df["BB_M"], df["BB_U"], df["BB_L"] = bollinger(close)
    df["ATR14"] = atr(df, 14)

    # S/R & S/D
    sup_levels, res_levels = sr_levels_from_swings(df)
    supply_zones, demand_zones = sd_zones_from_swings(df)

    # Nearest S/R
    last_close = close.iloc[-1]
    sup_near = max([lv for lv in sup_levels if lv <= last_close], default=np.nan)
    res_near = min([lv for lv in res_levels if lv >= last_close], default=np.nan)

    # Zone proximity
    in_supply = False
    in_demand = False
    for (_, zlow, zhigh) in supply_zones[-2:]:  # cek 2 zona terbaru
        if price_in_zone(last_close, zlow, zhigh):
            in_supply = True; break
    for (_, zlow, zhigh) in demand_zones[-2:]:
        if price_in_zone(last_close, zlow, zhigh):
            in_demand = True; break

    # Voting
    vote = 0
    # Trend
    if df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]: vote += 1
    elif df["EMA5"].iloc[-1] < df["EMA20"].iloc[-1]: vote -= 1
    # Momentum/mean reversion
    if df["RSI14"].iloc[-1] < 30: vote += 1
    elif df["RSI14"].iloc[-1] > 70: vote -= 1
    # MACD relative
    if df["MACD"].iloc[-1] > df["MACD_SIG"].iloc[-1]: vote += 1
    else: vote -= 1
    # Proximity to S/R (dalam 0.5 * ATR)
    atr_now = df["ATR14"].iloc[-1] if pd.notna(df["ATR14"].iloc[-1]) else (df["High"]-df["Low"]).tail(14).mean()
    if pd.notna(sup_near) and abs(last_close - sup_near) <= 0.5*atr_now:
        vote += 1
    if pd.notna(res_near) and abs(last_close - res_near) <= 0.5*atr_now:
        vote -= 1
    # Inside Supply/Demand zone
    if in_demand: vote += 1
    if in_supply: vote -= 1

    # Label
    if vote >= 2: label = "STRONG BUY"
    elif vote == 1: label = "BUY"
    elif vote == 0: label = "HOLD"
    elif vote == -1: label = "SELL"
    else: label = "STRONG SELL"

    extras = {
        "sup_levels": sup_levels, "res_levels": res_levels,
        "supply_zones": supply_zones, "demand_zones": demand_zones,
        "sup_near": sup_near, "res_near": res_near,
        "in_supply": in_supply, "in_demand": in_demand,
        "vote": vote, "label": label
    }
    return df, extras

# ----------------------------
# 5) Plot
# ----------------------------
def plot_chart(df: pd.DataFrame, extras, title: str):
    ap = [
        mpf.make_addplot(df["EMA5"], color="blue"),
        mpf.make_addplot(df["EMA20"], color="orange"),
        mpf.make_addplot(df["BB_U"], color="grey"),
        mpf.make_addplot(df["BB_M"], color="purple"),
        mpf.make_addplot(df["BB_L"], color="grey"),
    ]
    fig, axlist = mpf.plot(df, type="candle", style="charles", addplot=ap,
                           returnfig=True, figsize=(12,6), title=title)

    ax = axlist[0]
    # Garis S/R terdekat
    if not np.isnan(extras["sup_near"]):
        ax.axhline(extras["sup_near"], linestyle="--")
    if not np.isnan(extras["res_near"]):
        ax.axhline(extras["res_near"], linestyle="--")

    # Shade Supply/Demand zones (2 terakhir)
    for (_, zlow, zhigh) in extras["demand_zones"][-2:]:
        ax.axhspan(zlow, zhigh, alpha=0.15)
    for (_, zlow, zhigh) in extras["supply_zones"][-2:]:
        ax.axhspan(zlow, zhigh, alpha=0.15)

    st.pyplot(fig)

# ----------------------------
# 6) Ekspor CSV otomatis
# ----------------------------
def append_signal_csv(path, row_dict, header=True):
    df = pd.DataFrame([row_dict])
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=(header and not file_exists), index=False)

# ----------------------------
# 7) UI Streamlit
# ----------------------------
st.title("📈 Trading Signals (EMA + RSI + MACD + Bollinger + S/R + Supply/Demand)")

with st.sidebar:
    st.subheader("Pengaturan Data")
    symbols = st.multiselect("Pilih simbol (bisa >1)", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS)
    custom = st.text_input("Tambahkan ticker manual (opsional, pisahkan koma)", "")
    if custom.strip():
        for s in [x.strip() for x in custom.split(",") if x.strip()]:
            symbols.append(s)
    symbols = list(dict.fromkeys(symbols))  # unique & keep order

    interval = st.selectbox("Timeframe", INTERVALS, index=3)
    period = st.selectbox("Periode", PERIODS, index=2)
    st.caption("Catatan: yfinance membatasi beberapa kombinasi period/interval.")
    run = st.button("🚀 Hitung Sinyal")

tab_chart, tab_table = st.tabs(["📊 Chart & Detail", "📋 Ringkasan Multi-Pair"])

if run:
    summary_rows = []
    first_symbol_for_chart = None
    for sym in symbols:
        yahoo_t = map_ticker(sym)
        df = fetch_data(yahoo_t, period=period, interval=interval)
        if df.empty:
            st.warning(f"{sym} → data kosong. Coba interval/period lain.")
            continue

        df_sig, extras = compute_signals(df)
        last = df_sig.iloc[-1]
        row = {
            "timestamp": last.name, "symbol": sym, "ticker": yahoo_t,
            "close": float(last["Close"]),
            "ema5": float(last["EMA5"]), "ema20": float(last["EMA20"]),
            "rsi14": float(last["RSI14"]),
            "macd": float(last["MACD"]), "macd_sig": float(last["MACD_SIG"]),
            "support_near": extras["sup_near"], "resistance_near": extras["res_near"],
            "in_demand_zone": extras["in_demand"], "in_supply_zone": extras["in_supply"],
            "vote": extras["vote"], "signal": extras["label"]
        }
        summary_rows.append(row)

        # Ekspor otomatis (gabungan semua simbol)
        append_signal_csv("signals_all.csv", row, header=True)

        # Simpan simbol pertama yang sukses untuk chart
        if first_symbol_for_chart is None:
            first_symbol_for_chart = (sym, df_sig, extras)

    # Tabel Ringkasan
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        with tab_table:
            st.dataframe(df_summary.sort_values(["symbol", "timestamp"]).reset_index(drop=True))
            csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download ringkasan (CSV)", data=csv_bytes, file_name="signals_summary.csv", mime="text/csv")
        # Chart detail untuk simbol pertama
        with tab_chart:
            sym, df_sig, extras = first_symbol_for_chart
            st.subheader(f"Chart: {sym} • {interval} • Signal: {extras['label']} (vote={extras['vote']})")
            plot_chart(df_sig, extras, title=f"{sym} • {interval}")
            st.caption("Chart menampilkan EMA5/20, Bollinger Bands, garis S/R terdekat, dan shading zona Supply/Demand.")
        st.success("✅ Sinyal diekspor otomatis ke file: signals_all.csv")
    else:
        st.error("Tidak ada data yang bisa diproses.")
else:
    st.info("Pilih simbol, timeframe, lalu klik **Hitung Sinyal**.")
