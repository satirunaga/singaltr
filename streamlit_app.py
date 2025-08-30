import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trading Profit Analyzer", layout="wide")

st.title("📊 Trading Profit Analyzer")

st.write("Upload satu atau beberapa file laporan trading (format Excel / CSV).")

# Upload file
uploaded_files = st.file_uploader(
    "Pilih file laporan trading", type=["xlsx", "xls", "csv"], accept_multiple_files=True
)

def process_file(file):
    # Baca file sesuai format
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=0)

    # Deteksi header (baris pertama biasanya metadata)
    if "Profit" not in df.columns:
        df = pd.read_excel(file, sheet_name=0, header=1)

    # Ambil kolom yang dibutuhkan
    df = df[["Time", "Profit"]].dropna()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Date"] = df["Time"].dt.date
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

    # Hitung profit per hari
    per_day = df.groupby("Date", as_index=False)["Profit"].sum()

    # Hitung hasil utama
    max_row = per_day.loc[per_day["Profit"].idxmax()]
    max_profit = max_row["Profit"]
    max_date = max_row["Date"]
    total_profit = per_day["Profit"].sum()
    percentage = (max_profit / total_profit) * 100 if total_profit != 0 else 0

    return per_day, max_profit, max_date, total_profit, percentage

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📁 Hasil untuk file: `{file.name}`")

        try:
            per_day, max_profit, max_date, total_profit, percentage = process_file(file)

            st.write("### Profit per Hari")
            st.dataframe(per_day)

            st.metric("🔥 Profit terbesar", f"{max_profit:,.2f}", delta=f"pada {max_date}")
            st.metric("💰 Total Profit", f"{total_profit:,.2f}")
            st.metric("📈 Kontribusi Profit Terbesar", f"{percentage:.2f}%")
        except Exception as e:
            st.error(f"Gagal memproses {file.name}: {e}")
