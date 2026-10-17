import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# ==========================================
# 0. 帳號密碼登入驗證保護機制
# ==========================================
def check_password():
    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## 🔐 系統登入驗證")
    st.info("請輸入您的專屬帳號與密碼以存取系統。")
    
    with st.form("login_form"):
        username = st.text_input("帳號 (Username)")
        password = st.text_input("密碼 (Password)", type="password")
        submit_btn = st.form_submit_button("登入")

    if submit_btn:
        passwords_dict = st.secrets.get("passwords", {})
        if username in passwords_dict and passwords_dict[username] == password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ 帳號或密碼錯誤，請重新輸入。")
            
    return False

if not check_password():
    st.stop()

# ==========================================
# 1. 介面與極簡回測邏輯
# ==========================================
st.set_page_config(page_title="極簡診斷版系統", layout="wide")
st.title("🛠️ 系統極簡診斷與測試版")
st.caption("此版本用於測試基礎連線與運算速度，排查卡頓原因。")

if st.sidebar.button("🔒 登出系統"):
    st.session_state["password_correct"] = False
    st.rerun()

st.sidebar.header("設定")
run_test = st.sidebar.button("🚀 執行極簡測試")

@st.cache_data(ttl=86400)
def load_data():
    tickers = {'S&P500': '^GSPC', 'Treasury': 'TLT'}
    df_raw = yf.download(list(tickers.values()), start="2015-01-01", progress=False, timeout=10)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_prices = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns else df_raw['Close']
    else:
        df_prices = df_raw[['Close']]
    df_prices = df_prices.rename(columns={v: k for k, v in tickers.items()})
    return df_prices.dropna()

with st.spinner("正在下載測試資料..."):
    prices = load_data()

if prices.empty:
    st.error("❌ 無法下載 Yahoo Finance 資料，可能遭受 IP 限制或網路逾時。")
    st.stop()

st.success("✅ 資料下載成功！")
st.dataframe(prices.tail())

if run_test:
    st.info("正在執行極速運算...")
    sp = prices['S&P500']
    sma = sp.rolling(50).mean()
    returns = sp.pct_change()
    
    strategy_ret = returns * (sp > sma).shift(1)
    cum_perf = (1 + strategy_ret.fillna(0)).cumprod()
    
    st.line_chart(cum_perf)
    st.success("測試運算完成！")
else:
    st.info("👈 請點擊左側『執行極簡測試』按鈕。")
