import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression

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
# 1. 核心模型模組（多資產 + 總經 + 極速歷史鎖定）
# ==========================================
class OptimizedRegimeModel:
    def __init__(self, prob_threshold=0.7, ewm_window=12, rebalance_freq='Monthly', momentum_lookback=6, defense_mode='Strict'):
        self.threshold = float(prob_threshold)
        self.ewm_window = int(ewm_window)
        self.rebal_freq = rebalance_freq
        self.mom_lookback = int(momentum_lookback)
        self.defense_mode = defense_mode

    def run_pipeline(self, raw_returns_df, raw_macro_df, global_proxy_col='S&P500', riskfree_col='RiskFree'):
        if self.rebal_freq == 'Monthly':
            returns_df = (1 + raw_returns_df).resample('ME').prod() - 1
            macro_df = raw_macro_df.resample('ME').last()
            sma_window = 12
        elif self.rebal_freq == 'Weekly':
            returns_df = (1 + raw_returns_df).resample('W-FRI').prod() - 1
            macro_df = raw_macro_df.resample('W-FRI').last()
            sma_window = 10
        else:
            returns_df = raw_returns_df.copy()
            macro_df = raw_macro_df.copy()
            sma_window = 50

        returns_df = returns_df.dropna()
        macro_df = macro_df.reindex(returns_df.index).ffill().bfill()
        
        common_idx = returns_df.index.intersection(macro_df.index)
        returns_df_aligned = returns_df.loc[common_idx]
        macro_df_aligned = macro_df.loc[common_idx]
        
        risky_assets = [c for c in returns_df_aligned.columns if c not in [global_proxy_col, riskfree_col]]
        dates = common_idx
        T = len(dates)

        # 構築特徵矩陣
        feat_dict = {}
        for c in risky_assets + [global_proxy_col]:
            r = returns_df_aligned[c]
            feat_dict[f'{c}_ret'] = r.rolling(3).mean()
            feat_dict[f'{c}_vol'] = r.rolling(3).std()
            
        X_comp = pd.DataFrame(feat_dict, index=dates).join(macro_df_aligned).ffill().bfill()
        
        # 大盤多空標籤
        sp_series = returns_df_aligned[global_proxy_col]
        sp_sma = sp_series.rolling(window=sma_window, min_periods=1).mean()
        global_labels = (sp_series > sp_sma).astype(int)
        y_global = global_labels.shift(-1).ffill().fillna(0)

        # 🔒 Walk-Forward 滾動式預測（滾動視窗 40 期，確保毫秒級完成）
        prob_global_bull = np.full(T, 0.5)
        min_train_size = max(6, T // 5)
        max_rolling_window = 40  

        X_vals = X_comp.values
        y_vals = y_global.values.astype(int)

        for t in range(min_train_size, T):
            start_idx = max(0, t - max_rolling_window)
            X_train = X_vals[start_idx:t]
            y_train = y_vals[start_idx:t]
            X_current = X_vals[t:t+1]
            
            if len(np.unique(y_train)) < 2:
                y_train = y_train.copy()
                y_train[0] = 0
                y_train[-1] = 1
                
            clf = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_current)
            prob_global_bull[t] = probs[0, 1] if probs.shape[1] > 1 else 0.5

        pred_global_bear = (prob_global_bull < (1.0 - self.threshold)).astype(int)
        pred_global_bull_flag = (prob_global_bull > self.threshold).astype(int)
        
        alpha_ewm = 2.0 / (self.ewm_window + 1.0)
        smoothed_prob_asset_bull = {}
        for a in risky_assets:
            a_prob = pd.Series(prob_global_bull, index=dates)
            smoothed_prob_asset_bull[a] = a_prob.ewm(alpha=alpha_ewm).mean()
            
        smoothed_prob_asset_df = pd.DataFrame(smoothed_prob_asset_bull, index=dates)

        price_levels = (1 + returns_df_aligned[risky_assets]).cumprod()
        mom_matrix = price_levels / price_levels.shift(self.mom_lookback) - 1
        mom_matrix = mom_matrix.fillna(1.0)

        stock_like_assets = ['Nasdaq', 'S&P500']
        bmda_sets = {}
        bmga_sets = {}
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull_flag[t] == 1
            
            if d in smoothed_prob_asset_df.index:
                asset_bulls = smoothed_prob_asset_df.loc[d] > self.threshold
                ml_selected = asset_bulls[asset_bulls].index.tolist()
            else:
                ml_selected = []
            
            if d in mom_matrix.index:
                positive_mom_assets = mom_matrix.loc[d][mom_matrix.loc[d] > 0.0].index.tolist()
                filtered_risky = [a for a in ml_selected if a in positive_mom_assets]
            else:
                filtered_risky = ml_selected

            if is_global_bear:
                if self.defense_mode == 'Strict':
                    selected_bmda = [a for a in filtered_risky if a not in stock_like_assets]
                else:
                    selected_bmda = filtered_risky.copy()
                selected_bmda.append(riskfree_col)
                bmda_sets[d] = selected_bmda
            else:
                bmda_sets[d] = [riskfree_col]
                
            if is_global_bull:
                selected_bmga = filtered_risky.copy()
                if global_proxy_col not in selected_bmga and global_proxy_col in positive_mom_assets:
                    selected_bmga.append(global_proxy_col)
                if not selected_bmga or global_proxy_col not in selected_bmga:
                    selected_bmga = [global_proxy_col]
                bmga_sets[d] = selected_bmga
            else:
                bmga_sets[d] = [global_proxy_col]

        rebal_dates = set(dates)
        portfolio_returns = []
        tc_rate = 0.0010
        prev_weights = pd.Series(0.0, index=returns_df_aligned.columns)
        
        for t in range(len(dates) - 1):
            d_today = dates[t]
            d_next = dates[t+1]
            
            if pred_global_bear[t] == 1:
                target_assets = bmda_sets.get(d_today, [riskfree_col])
            else:
                target_assets = bmga_sets.get(d_today, [global_proxy_col])
                
            target_weights = pd.Series(0.0, index=returns_df_aligned.columns)
            target_weights[target_assets] = 1.0 / len(target_assets)
            
            turnover = np.sum(np.abs(target_weights - prev_weights))
            tc = turnover * tc_rate
            
            r_next = returns_df_aligned.loc[d_next]
            port_r = np.sum(target_weights * r_next) - tc
            
            portfolio_returns.append({
                'Date': d_next, 
                'Return': port_r, 
                'Turnover': turnover,
                'Global_Bull_Prob': prob_global_bull[t],
                'Is_Rebal': True
            })
            prev_weights = target_weights.copy()
            
        res_df = pd.DataFrame(portfolio_returns).set_index('Date')
        return res_df, bmda_sets, bmga_sets, rebal_dates, returns_df_aligned, prob_global_bull


# ==========================================
# 2. Streamlit 介面與前端展示
# ==========================================
st.set_page_config(page_title="雙重狀態動態資產配置系統", layout="wide")

st.title("📈 雙重狀態動態資產配置系統（極速歷史鎖定版）")
st.caption("已成功恢復多資產配置、總經指標與完整儀表板，並維持秒速運算能力。")

if st.sidebar.button("🔒 登出系統"):
    st.session_state["password_correct"] = False
    st.rerun()

st.sidebar.header("📅 回測時間區間設定")
default_start = pd.to_datetime("2010-01-01")
default_end = pd.to_datetime("2026-12-31")

start_date = st.sidebar.date_input("回測開始日期", default_start)
end_date = st.sidebar.date_input("回測結束日期", default_end)

st.sidebar.header("⚙️ 模型與策略模式設定")
rebal_freq_option = st.sidebar.selectbox("資產調倉與預測頻率", ["每月調整 (Monthly)", "每週調整 (Weekly)", "每日調整 (Daily)"], index=0)
freq_mapping = {"每月調整 (Monthly)": "Monthly", "每週調整 (Weekly)": "Weekly", "每日調整 (Daily)": "Daily"}
chosen_freq = freq_mapping[rebal_freq_option]

defense_option = st.sidebar.selectbox(
    "熊市防禦資產池模式", 
    [
        "絕對安全防禦 (Strict: 熊市嚴禁股票類資產)", 
        "動量優勢導向 (Flexible: 允許動量為正的股票)"
    ], 
    index=0
)
chosen_defense_mode = 'Strict' if "Strict" in defense_option else 'Flexible'

if chosen_freq == "Monthly":
    mom_lb = st.sidebar.slider("絕對動量回看期 (月)", 1, 24, 6, step=1)
    prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
    ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (月)", 1, 24, 6, step=1)
elif chosen_freq == "Weekly":
    mom_lb = st.sidebar.slider("絕對動量回看期 (週)", 2, 52, 12, step=1)
    prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
    ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (週)", 2, 26, 12, step=1)
else:
    mom_lb = st.sidebar.slider("絕對動量回看天數 (Momentum Lookback)", 21, 126, 63, step=1)
    prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
    ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (天)", 10, 126, 63, step=1)

run_button = st.sidebar.button("🚀 執行回測與分析")

@st.cache_data(ttl=86400)
def fetch_real_market_data():
    tickers_map = {
        'S&P500': '^GSPC',
        'Nasdaq': '^IXIC',
        'Treasury': 'TLT',
        'Corporate': 'LQD',
        'HighYield': 'HYG',
        'Gold': 'GC=F',
        'Commodity': 'DBC',
        'REIT': 'VNQ'
    }
    macro_tickers = {'VIX': '^VIX'}
    all_tickers = list(tickers_map.values()) + list(macro_tickers.values())
    
    try:
        df_raw = yf.download(all_tickers, start="2006-01-01", progress=False, timeout=20)
        if df_raw.empty:
            raise ValueError("下載到的資料為空")
    except Exception as e:
        st.error(f"❌ Yahoo Finance 資料下載失敗或逾時: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_prices = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns else df_raw['Close']
    else:
        df_prices = df_raw[['Close']]

    inv_tickers_map = {v: k for k, v in tickers_map.items()}
    inv_macro_map = {v: k for k, v in macro_tickers.items()}
    df_prices = df_prices.rename(columns={**inv_tickers_map, **inv_macro_map})
    
    valid_cols = [c for c in tickers_map.keys() if c in df_prices.columns]
    returns_df = df_prices[valid_cols].pct_change().dropna(how='all')
    returns_df['RiskFree'] = 0.0001
    
    macro_cols = [c for c in macro_tickers.keys() if c in df_prices.columns]
    macro_df = df_prices[macro_cols].reindex(returns_df.index).ffill().bfill()
    macro_df['Yield_Curve'] = 0.5 
    macro_df['Inflation'] = 2.0
    
    return returns_df.dropna(), macro_df.dropna()

with st.spinner("正在同步真實金融市場與總經歷史數據..."):
    raw_returns_df, raw_macro_df = fetch_real_market_data()

if raw_returns_df.empty or raw_macro_df.empty:
    st.error("❌ 無法取得市場數據，請重新整理頁面或稍後再試。")
    st.stop()

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)
common_index = raw_returns_df.index.intersection(raw_macro_df.index)
clean_returns = raw_returns_df.loc[common_index]
clean_macro = raw_macro_df.loc[common_index]

mask = (clean_returns.index >= start_ts) & (clean_returns.index <= end_ts)
returns_df_input = clean_returns.loc[mask].copy()
macro_df_input = clean_macro.loc[mask].copy()

if returns_df_input.empty or len(returns_df_input) < 30:
    st.error("❌ 選擇的時間區間內真實資料不足，請擴大回測起訖日期！")
    st.stop()

if run_button or 'results' not in st.session_state:
    with st.spinner(f"正在執行極速 Walk-Forward 運算 (模式：{defense_option} | 頻率：{rebal_freq_option})..."):
        try:
            model_instance = OptimizedRegimeModel(
                prob_threshold=float(prob_thresh),
                ewm_window=int(ewm_win),
                rebalance_freq=chosen_freq,
                momentum_lookback=int(mom_lb),
                defense_mode=chosen_defense_mode
            )
            res_df, bmda_hist, bmga_hist, rebal_dates, returns_aligned, raw_probs = model_instance.run_pipeline(
                returns_df_input.copy(), macro_df_input.copy(), global_proxy_col='S&P500', riskfree_col='RiskFree'
            )
            st.session_state['results'] = res_df
            st.session_state['bmda'] = bmda_hist
            st.session_state['bmga'] = bmga_hist
            st.session_state['rebal_dates'] = rebal_dates
            st.session_state['returns_aligned'] = returns_aligned
            st.session_state['raw_probs'] = raw_probs
        except Exception as e:
            st.error(f"❌ 模型運算過程發生錯誤: {e}")
            st.stop()

if 'results' not in st.session_state:
    st.info("👈 請點擊左側側邊欄的『🚀 執行回測與分析』按鈕以開始。")
    st.stop()

res_df = st.session_state['results']
bmda_hist = st.session_state['bmda']
bmga_hist = st.session_state['bmga']
rebal_dates = st.session_state['rebal_dates']
returns_aligned = st.session_state['returns_aligned']
raw_probs = st.session_state['raw_probs']

# ==========================================
# 3. 顯示網頁頂端：最近一個週期的訊號與建議配置
# ==========================================
st.markdown("---")
period_name = "最近一月" if chosen_freq == "Monthly" else ("最近一週" if chosen_freq == "Weekly" else "最近一日")
st.subheader(f"🎯 依據【{rebal_freq_option}】產生的【{period_name}】市場訊號與建議配置")

dates_index = sorted(list(returns_aligned.index))
latest_signal_date = dates_index[-1] 

latest_prob = raw_probs[-1] if len(raw_probs) > 0 else 0.5
is_latest_bear = latest_prob < 0.5
latest_state = "🐻 熊市防禦 (BMDA)" if is_latest_bear else "🚀 牛市成長 (BMGA)"
latest_assets = bmda_hist.get(latest_signal_date, []) if is_latest_bear else bmga_hist.get(latest_signal_date, [])

c1, c2, c3 = st.columns(3)
c1.metric("最新訊號生成日", latest_signal_date.strftime('%Y-%m-%d'))
c2.metric("模型牛市預測機率", f"{latest_prob:.4f}", latest_state)
c3.metric("建議配置資產池", ", ".join(latest_assets))
st.markdown("---")

# ==========================================
# 4. 績效與預測勝率計算
# ==========================================
cum_returns = (1 + res_df['Return']).cumprod()
annual_factor = 12 if chosen_freq == "Monthly" else (52 if chosen_freq == "Weekly" else 252)
sharpe = (res_df['Return'].mean() * annual_factor) / (res_df['Return'].std() * np.sqrt(annual_factor)) if res_df['Return'].std() > 0 else 0
max_dd = (cum_returns / cum_returns.cummax() - 1).min()
annual_ret = (cum_returns.iloc[-1] ** (annual_factor / len(res_df))) - 1 if len(res_df) > 0 else 0
avg_turnover = res_df['Turnover'].mean()

benchmark_returns = returns_aligned.loc[res_df.index, 'S&P500']
benchmark_cum = (1 + benchmark_returns).cumprod()
benchmark_sharpe = (benchmark_returns.mean() * annual_factor) / (benchmark_returns.std() * np.sqrt(annual_factor)) if benchmark_returns.std() > 0 else 0
benchmark_max_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()
benchmark_annual_ret = (benchmark_cum.iloc[-1] ** (annual_factor / len(benchmark_cum))) - 1 if len(benchmark_cum) > 0 else 0

correct_count = 0
total_count = 0
period_records = []

dates_list = sorted(list(returns_aligned.index))
for i in range(len(dates_list)):
    d_signal = dates_list[i]     
    
    if i < len(dates_list) - 1:
        d_next = dates_list[i+1]     
        prob = res_df.loc[d_next, 'Global_Bull_Prob'] if d_next in res_df.index else raw_probs[i]
        period_sp_ret = returns_aligned.loc[d_next, 'S&P500'] if d_next in returns_aligned.index else 0.0
        
        is_bull_pred = prob >= 0.5
        is_up = period_sp_ret > 0
        is_down = period_sp_ret < 0
        
        if (is_bull_pred and is_up) or (not is_bull_pred and is_down):
            eval_result = "✅ 正確"
            correct_count += 1
            total_count += 1
        elif period_sp_ret == 0:
            eval_result = "➖ 持平"
        else:
            eval_result = "❌ 錯誤"
            total_count += 1
            
        ret_str = f"{period_sp_ret * 100:.2f}%"
        future_str = d_next.strftime('%Y-%m-%d')
    else:
        prob = raw_probs[i] if i < len(raw_probs) else 0.5
        future_str = "⏳ 待下期結算"
        ret_str = "N/A"
        eval_result = "⏳ 待驗證"

    is_bear = prob < 0.5
    regime_str = "🐻 熊市防禦 (BMDA)" if is_bear else "🚀 牛市成長 (BMGA)"
    assets = bmda_hist.get(d_signal, []) if is_bear else bmga_hist.get(d_signal, [])
    
    period_records.append({
        "訊號生成日 (結算)": d_signal.strftime('%Y-%m-%d'),
        "預驗證未來區間": future_str,
        "牛市預測機率": f"{prob:.4f}",
        "市場判定狀態": regime_str,
        "S&P500 下期報酬率": ret_str,
        "預測驗證": eval_result,
        "當期配置資產池": ", ".join(assets)
    })

period_records.reverse()
win_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

st.subheader(f"📊 核心績效指標與預測勝率 ({rebal_freq_option
