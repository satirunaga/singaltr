# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Signals: EMA + S/R + S/D", layout="wide")

# ----------------------------
# 0) Utility: data & mapping
# ----------------------------
TICKER_MAP = {
    # Forex
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X", "USDCHF": "CHF=X", "USDCAD": "CAD=X",
    "NZDUSD": "NZD=X", "XAUUSD": "XAUUSD=X", "XAGUSD": "XAGUSD=X",
    # Crypto
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "LTCUSD": "LTC-USD",
    # Stocks (contoh)
    "AAPL": "AAPL", "TSLA": "TSLA", "MSFT": "MSFT"
}
DEFAULT_SYMBOLS = ["BTCUSD", "EURUSD", "XAUUSD"]

INTERVALS = ["1m","5m","15m","1h","1d"]
PERIODS   = ["1d","5d","1mo","3mo","6mo","1y"]

# Batasan yfinance
VALID_COMBOS = {
    "1m": ["1d","5d","7d"],
    "5m": ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h": ["1d","5d","1mo","3mo"],
    "1d": ["1mo","3mo","6mo","1y","5y","10y","max"]
}

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
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    ma_up, ma_down = up.ewm(alpha=1/period, adjust=False).mean(), down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_, slow_ = series.ewm(span=fast, adjust=False).mean(), series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ - slow_; sig = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig, macd_line - sig

def bollinger(series, length=20, mult=2):
    mb, sd = series.rolling(length).mean(), series.rolling(length).std()
    return mb, mb+mult*sd, mb-mult*sd

def atr(df, n=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ----------------------------
# 2) Swing High/Low & S/R
# ----------------------------
def find_swings(df, left=3, right=3):
    win = left + right + 1
    highs = df["High"].rolling(win, center=True).apply(lambda x: 1.0 if np.argmax(x)==left else 0.0, raw=True).fillna(0).astype(bool)
    lows  = df["Low"].rolling(win, center=True).apply(lambda x: 1.0 if np.argmin(x)==left else 0.0, raw=True).fillna(0).astype(bool)
    return highs, lows

def cluster_levels(prices, tol):
    if len(prices)==0: return []
    levels = sorted(prices.values); clustered=[]
    for p in levels:
        if not clustered: clustered.append(p)
        elif abs(p-clustered[-1]) <= tol: clustered[-1] = (clustered[-1]+p)/2
        else: clustered.append(p)
    return clustered

def sr_levels_from_swings(df, left=3, right=3, tol_mult=0.5):
    highs,lows = find_swings(df,left,right)
    swing_highs, swing_lows = df.loc[highs,"High"], df.loc[lows,"Low"]
    _atr = atr(df,14).dropna()
    tol = (_atr.iloc[-1] if len(_atr)>0 else (df["High"]-df["Low"]).tail(14).mean())*tol_mult
    return cluster_levels(swing_lows,tol), cluster_levels(swing_highs,tol)

# ----------------------------
# 3) Supply/Demand Zones
# ----------------------------
def sd_zones_from_swings(df, atr_mult=1.0, max_zones=4):
    highs,lows = find_swings(df)
    swings_hi, swings_lo = df.loc[highs,["High"]].tail(max_zones), df.loc[lows,["Low"]].tail(max_zones)
    _atr, last_atr = atr(df,14), None
    last_atr = _atr.iloc[-1] if _atr.notna().any() else (df["High"]-df["Low"]).tail(14).mean()
    supply=[(i, float(r["High"])-atr_mult*last_atr, float(r["High"])) for i,r in swings_hi.iterrows()]
    demand=[(i, float(r["Low"]), float(r["Low"])+atr_mult*last_atr) for i,r in swings_lo.iterrows()]
    return supply,demand

def price_in_zone(price, zl, zh): return zl<=price<=zh

# ----------------------------
# 4) Voting & Signal
# ----------------------------
def compute_signals(df):
    c=df["Close"]
    df["EMA5"],df["EMA20"]=ema(c,5),ema(c,20)
    df["RSI14"]=rsi(c,14)
    df["MACD"],df["MACD_SIG"],df["MACD_HIST"]=macd(c)
    df["BB_M"],df["BB_U"],df["BB_L"]=bollinger(c)
    df["ATR14"]=atr(df,14)

    sup,res=sr_levels_from_swings(df)
    sup_z,res_z=sd_zones_from_swings(df)
    lc=c.iloc[-1]
    sup_n=max([lv for lv in sup if lv<=lc],default=np.nan)
    res_n=min([lv for lv in res if lv>=lc],default=np.nan)

    in_sup=any(price_in_zone(lc,zl,zh) for _,zl,zh in res_z[-2:])
    in_dem=any(price_in_zone(lc,zl,zh) for _,zl,zh in sup_z[-2:])

    v=0
    v += 1 if df["EMA5"].iloc[-1]>df["EMA20"].iloc[-1] else -1
    v += 1 if df["RSI14"].iloc[-1]<30 else -1 if df["RSI14"].iloc[-1]>70 else 0
    v += 1 if df["MACD"].iloc[-1]>df["MACD_SIG"].iloc[-1] else -1
    atrn=df["ATR14"].iloc[-1] if pd.notna(df["ATR14"].iloc[-1]) else (df["High"]-df["Low"]).tail(14).mean()
    if pd.notna(sup_n) and abs(lc-sup_n)<=0.5*atrn: v+=1
    if pd.notna(res_n) and abs(lc-res_n)<=0.5*atrn: v-=1
    if in_dem: v+=1
    if in_sup: v-=1

    lbl="STRONG BUY" if v>=2 else "BUY" if v==1 else "HOLD" if v==0 else "SELL" if v==-1 else "STRONG SELL"

    return df,{"sup_levels":sup,"res_levels":res,"supply_zones":res_z,"demand_zones":sup_z,
               "sup_near":sup_n,"res_near":res_n,"in_supply":in_sup,"in_demand":in_dem,"vote":v,"label":lbl}

# ----------------------------
# 5) Plot
# ----------------------------
def plot_chart(df,extras,title):
    ap=[mpf.make_addplot(df["EMA5"],color="blue"),mpf.make_addplot(df["EMA20"],color="orange"),
        mpf.make_addplot(df["BB_U"],color="grey"),mpf.make_addplot(df["BB_M"],color="purple"),mpf.make_addplot(df["BB_L"],color="grey")]
    fig,axlist=mpf.plot(df,type="candle",style="charles",addplot=ap,returnfig=True,figsize=(12,6),title=title)
    ax=axlist[0]
    if not np.isnan(extras["sup_near"]): ax.axhline(extras["sup_near"],ls="--")
    if not np.isnan(extras["res_near"]): ax.axhline(extras["res_near"],ls="--")
    for _,zl,zh in extras["demand_zones"][-2:]: ax.axhspan(zl,zh,alpha=0.15)
    for _,zl,zh in extras["supply_zones"][-2:]: ax.axhspan(zl,zh,alpha=0.15)
    st.pyplot(fig)

# ----------------------------
# 6) Ekspor CSV
# ----------------------------
def append_signal_csv(path,row,header=True):
    df=pd.DataFrame([row]); df.to_csv(path,mode="a",header=(header and not os.path.exists(path)),index=False)

# ----------------------------
# 7) UI
# ----------------------------
st.title("📈 Trading Signals (EMA + RSI + MACD + Bollinger + S/R + Supply/Demand)")

with st.sidebar:
    st.subheader("Pengaturan Data")
    syms=st.multiselect("Pilih simbol",DEFAULT_SYMBOLS,default=DEFAULT_SYMBOLS)
    cust=st.text_input("Tambahkan ticker manual (pisahkan koma)","")
    if cust.strip(): syms+=[s.strip() for s in cust.split(",") if s.strip()]
    syms=list(dict.fromkeys(syms))
    interval=st.selectbox("Timeframe",INTERVALS,index=3)
    period=st.selectbox("Periode",PERIODS,index=2)

    # Validasi period-interval
    if period not in VALID_COMBOS.get(interval,[]):
        st.warning(f"Kombinasi {interval}+{period} tidak valid. Diganti ke {VALID_COMBOS[interval][-1]}.")
        period=VALID_COMBOS[interval][-1]

    run=st.button("🚀 Hitung Sinyal")

tab_chart,tab_table=st.tabs(["📊 Chart & Detail","📋 Ringkasan Multi-Pair"])

if run:
    summary=[]; first=None
    for s in syms:
        yt=map_ticker(s); df=fetch_data(yt,period,interval)
        if df.empty: st.warning(f"{s} → data kosong."); continue
        dfs,ext=compute_signals(df); last=dfs.iloc[-1]
        row={"timestamp":last.name,"symbol":s,"ticker":yt,"close":float(last["Close"]),
             "ema5":float(last["EMA5"]),"ema20":float(last["EMA20"]),"rsi14":float(last["RSI14"]),
             "macd":float(last["MACD"]),"macd_sig":float(last["MACD_SIG"]),
             "support_near":ext["sup_near"],"resistance_near":ext["res_near"],
             "in_demand_zone":ext["in_demand"],"in_supply_zone":ext["in_supply"],
             "vote":ext["vote"],"signal":ext["label"]}
        summary.append(row); append_signal_csv("signals_all.csv",row)
        if first is None: first=(s,dfs,ext)
    if summary:
        df_sum=pd.DataFrame(summary)
        with tab_table:
            st.dataframe(df_sum.sort_values(["symbol","timestamp"]).reset_index(drop=True))
            st.download_button("⬇️ Download CSV",data=df_sum.to_csv(index=False).encode("utf-8"),
                               file_name="signals_summary.csv",mime="text/csv")
        with tab_chart:
            s,dfs,ext=first
            st.subheader(f"{s} • {interval} • Signal: {ext['label']} (vote={ext['vote']})")
            plot_chart(dfs,ext,f"{s} • {interval}")
        st.success("✅ Sinyal tersimpan ke signals_all.csv")
    else: st.error("Tidak ada data.")
else:
    st.info("Pilih simbol lalu klik **Hitung Sinyal**.")
