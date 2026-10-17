import streamlit as st

# ==========================================
# 0. 帳號密碼登入驗證保護機制
# ==========================================
def check_password():
    """傳回 True 代表通過驗證；False 代表尚未驗證或失敗。"""
    
    # 檢查是否已經在 session_state 記錄為已登入
    if st.session_state.get("password_correct", False):
        return True

    # 建立一個好看的登入表單
    st.markdown("## 🔐 系統登入驗證")
    st.info("請輸入您的專屬帳號與密碼以存取美國公債預測與策略系統。")
    
    with st.form("login_form"):
        username = st.text_input("帳號 (Username)")
        password = st.text_input("密碼 (Password)", type="password")
        submit_btn = st.form_submit_button("登入")

    if submit_btn:
        # 從 st.secrets 中讀取設定好的密碼字典
        passwords_dict = st.secrets.get("passwords", {})
        
        if username in passwords_dict and passwords_dict[username] == password:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = username
            st.rerun()  # 驗證成功後重新整理頁面進入主系統
        else:
            st.error("❌ 帳號或密碼錯誤，請重新輸入。")
            
    return False

# 執行登入檢查：如果沒通過，就直接停止執行後續程式碼
if not check_password():
    st.stop()

# ==========================================
# 接下來放您原本的主程式碼...
# ==========================================
st.success(f"歡迎回來，{st.session_state.get('logged_in_user')}！")

# 這裡可以加上一個「登出」按鈕方便切換
if st.sidebar.button("🔒 登出系統"):
    st.session_state["password_correct"] = False
    st.rerun()

# (您原本的 import、側邊欄參數、模型運算、圖表顯示等程式碼接在這邊...)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    USE_XGB = False

# ==========================================
# 1. 核心模型模組
# ==========================================
class StatisticalJumpModel:
    def __init__(self, n_clusters=2, jump_penalty=10.0):
        self.n_clusters = n_clusters
        self.jump_penalty = float(jump_penalty)
        self.centroids = None

    def fit_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        T, d = X.shape
        K = self.n_clusters
        
        np.random.seed(42)
        idx = np.random.choice(T, K, replace=False)
        centroids = X[idx].copy()
        prev_states = np.zeros(T, dtype=int)
        
        for iteration in range(20):
            emission_cost = cdist(X, centroids, metric='sqeuclidean')
            dp = np.zeros((T, K))
            pointers = np.zeros((T, K), dtype=int)
            dp[0] = emission_cost[0]
            
            for t in range(1, T):
                for k in range(K):
                    transition_costs = dp[t-1] + self.jump_penalty * (np.arange(K) != k)
                    best_prev = np.argmin(transition_costs)
                    dp[t, k] = transition_costs[best_prev] + emission_cost[t, k]
                    pointers[t, k] = best_prev
            
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmin(dp[-1])
            for t in range(T - 2, -1, -1):
                states[t] = pointers[t + 1, states[t + 1]]
                
            if np.array_equal(states, prev_states):
                break
            prev_states = states.copy()
            
            for k in range(K):
                if np.sum(states == k) > 0:
                    centroids[k] = np.mean(X[states == k], axis=0)
                    
        self.centroids = centroids
        return states


def extract_minimalist_features(returns_df, riskfree_col='RiskFree', hl_list=[1, 5, 10, 21]):
    risky_cols = [c for c in returns_df.columns if c != riskfree_col]
    features = {}
    for col in risky_cols:
        r = returns_df[col]
        for hl in hl_list:
            alpha = 1 - np.exp(-np.log(2) / hl)
            mean_r = r.ewm(alpha=alpha).mean()
            downside_r = r.clip(upper=0)
            downside_std = np.sqrt((downside_r**2).ewm(alpha=alpha).mean())
            sortino = mean_r / (downside_std + 1e-6)
            
            features[f'{col}_mean_{hl}'] = mean_r
            features[f'{col}_downside_{hl}'] = downside_std
            features[f'{col}_sortino_{hl}'] = sortino
            
    df_feat = pd.DataFrame(features, index=returns_df.index).dropna()
    df_feat = df_feat.loc[:, df_feat.var() > 1e-8]
    
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df_feat.values)
    return pd.DataFrame(scaled_vals, index=df_feat.index, columns=df_feat.columns)


class DualRegimeAllocationModel:
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=12, rebalance_freq='Monthly', momentum_lookback=6, defense_mode='Strict'):
        self.jp_global = float(jump_penalty_global)
        self.jp_asset = float(jump_penalty_asset)
        self.threshold = float(prob_threshold)
        self.ewm_window = int(ewm_window)
        self.rebal_freq = rebalance_freq
        self.mom_lookback = int(momentum_lookback)
        self.defense_mode = defense_mode
        
    def _create_classifier(self):
        if USE_XGB:
            return XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
        else:
            return XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

    def _safe_fit_predict_proba(self, X_train, y_train, X_all):
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            y_train = y_train.copy()
            mid = len(y_train) // 2
            y_train[:mid] = 0
            y_train[mid:] = 1
            
        clf = self._create_classifier()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_all)
        return probs[:, 1] if probs.shape[1] > 1 else np.full(len(X_all), 0.5)

    def run_pipeline(self, raw_returns_df, raw_macro_df, global_proxy_col='S&P500', riskfree_col='RiskFree'):
        if self.rebal_freq == 'Monthly':
            returns_df = (1 + raw_returns_df).resample('ME').prod() - 1
            macro_df = raw_macro_df.resample('ME').last()
            hl_list = [1, 3, 6, 12]
            sma_window = 12
        elif self.rebal_freq == 'Weekly':
            returns_df = (1 + raw_returns_df).resample('W-FRI').prod() - 1
            macro_df = raw_macro_df.resample('W-FRI').last()
            hl_list = [1, 2, 4, 12]
            sma_window = 10
        else:
            returns_df = raw_returns_df.copy()
            macro_df = raw_macro_df.copy()
            hl_list = [1, 5, 10, 21]
            sma_window = 50

        returns_df = returns_df.dropna()
        macro_df = macro_df.reindex(returns_df.index).ffill().bfill()

        X_min = extract_minimalist_features(returns_df, riskfree_col=riskfree_col, hl_list=hl_list)
        X_comp = pd.concat([X_min, macro_df], axis=1).reindex(X_min.index).ffill().bfill()
        
        common_idx = X_comp.index.intersection(returns_df.index)
        X_comp = X_comp.loc[common_idx]
        returns_df_aligned = returns_df.loc[common_idx]
        
        risky_assets = [c for c in returns_df_aligned.columns if c not in [global_proxy_col, riskfree_col]]
        dates = common_idx
        
        global_min_feat = [c for c in X_comp.columns if global_proxy_col in c]
        if not global_min_feat:
            global_min_feat = list(X_comp.columns[:3])
            
        sjm_global = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_global)
        global_states = sjm_global.fit_predict(X_comp[global_min_feat].values)
        
        sp_series = returns_df_aligned.loc[dates, global_proxy_col]
        sp_sma = sp_series.rolling(window=sma_window, min_periods=1).mean()
        trend_label = (sp_series > sp_sma).astype(int).values
        
        ret_g0 = sp_series[global_states == 0].mean()
        ret_g1 = sp_series[global_states == 1].mean()
        sjm_bull_state = 1 if ret_g1 > ret_g0 else 0
        sjm_label = (global_states == sjm_bull_state).astype(int)
        
        global_regimes_label = np.where((trend_label == 1) | (sjm_label == 1), 1, 0)
        if len(np.unique(global_regimes_label)) < 2:
            global_regimes_label = (sp_series > sp_series.median()).astype(int).values

        asset_regimes_label = {}
        for a in risky_assets:
            a_min_feat = [c for c in X_comp.columns if a in c]
            if not a_min_feat:
                continue
            sjm_asset = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_asset)
            a_states = sjm_asset.fit_predict(X_comp[a_min_feat].values)
            a_series = returns_df_aligned.loc[dates, a]
            a_sma = a_series.rolling(window=sma_window, min_periods=1).mean()
            a_trend = (a_series > a_sma).astype(int).values
            
            ret_a0 = a_series[a_states == 0].mean()
            ret_a1 = a_series[a_states == 1].mean()
            a_bull_state = 1 if ret_a1 > ret_a0 else 0
            a_sjm = (a_states == a_bull_state).astype(int)
            
            a_label = np.where((a_trend == 1) | (a_sjm == 1), 1, 0)
            if len(np.unique(a_label)) < 2:
                a_label = (a_series > a_series.median()).astype(int).values
            asset_regimes_label[a] = a_label
            
        asset_regimes_label_df = pd.DataFrame(asset_regimes_label, index=dates)

        y_global = pd.Series(global_regimes_label, index=dates).shift(-1).ffill().fillna(0)
        X_train_g = X_comp.iloc[:-1].values
        y_train_g = y_global.iloc[:-1].values.astype(int)
        
        prob_global_bull = self._safe_fit_predict_proba(X_train_g, y_train_g, X_comp.values)
        
        pred_global_bear = (prob_global_bull < (1.0 - self.threshold)).astype(int)
        pred_global_bull = (prob_global_bull > self.threshold).astype(int)
        
        alpha_ewm = 2.0 / (self.ewm_window + 1.0)
        smoothed_prob_asset_bull = {}
        
        for a in risky_assets:
            if a not in asset_regimes_label_df.columns:
                continue
            y_asset = asset_regimes_label_df[a].shift(-1).ffill().fillna(0)
            X_train_a = X_comp.iloc[:-1].values
            y_train_a = y_asset.iloc[:-1].values.astype(int)
            
            raw_prob_bull = self._safe_fit_predict_proba(X_train_a, y_train_a, X_comp.values)
            prob_series = pd.Series(raw_prob_bull, index=dates)
            smoothed_prob_asset_bull[a] = prob_series.ewm(alpha=alpha_ewm).mean()
            
        smoothed_prob_asset_df = pd.DataFrame(smoothed_prob_asset_bull)

        price_levels = (1 + returns_df_aligned[risky_assets]).cumprod()
        mom_matrix = price_levels / price_levels.shift(self.mom_lookback) - 1
        mom_matrix = mom_matrix.fillna(1.0)

        stock_like_assets = ['Nasdaq', 'S&P500']

        bmda_sets = {}
        bmga_sets = {}
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull[t] == 1
            
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
st.set_page_config(page_title="雙重狀態資產配置系統", layout="wide")

st.title("📈 雙重狀態動態資產配置系統（完整訊號顯示版）")
st.caption("結合 SJM、機器學習、絕對動量濾網，明細表已完整包含最新收盤訊號。")

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
    jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
    jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
    prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
    ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (月)", 1, 24, 6, step=1)
elif chosen_freq == "Weekly":
    mom_lb = st.sidebar.slider("絕對動量回看期 (週)", 2, 52, 12, step=1)
    jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
    jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
    prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
    ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (週)", 2, 26, 12, step=1)
else:
    mom_lb = st.sidebar.slider("絕對動量回看天數 (Momentum Lookback)", 21, 126, 63, step=1)
    jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
    jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
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
    
    df_raw = yf.download(all_tickers, start="2006-01-01", progress=False)
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
    with st.spinner(f"模型運算中 (模式：{defense_option} | 頻率：{rebal_freq_option})..."):
        model_instance = DualRegimeAllocationModel(
            jump_penalty_global=float(jp_global),
            jump_penalty_asset=float(jp_asset),
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
latest_signal_date = dates_index[-1] # 直接抓取最新一天（包含昨天）

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
# 4. 績效與預測勝率計算（包含最新未驗證訊號）
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
    d_signal = dates_list[i]     # 訊號生成日
    
    if i < len(dates_list) - 1:
        d_next = dates_list[i+1]     # 驗證報酬率的未來期
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
        # 最後一天（最新收盤日，如昨天），尚未有下期報酬可驗證
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

# 反轉列表讓最新的一筆（昨天）排在最上方
period_records.reverse()

win_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

st.subheader(f"📊 核心績效指標與預測勝率 ({rebal_freq_option})")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("年化夏普值", f"{sharpe:.2f}", f"大盤基準: {benchmark_sharpe:.2f}")
col2.metric("最大回撤", f"{max_dd * 100:.2f}%", f"大盤基準: {benchmark_max_dd * 100:.2f}%")
col3.metric("年化報酬率", f"{annual_ret * 100:.2f}%", f"大盤基準: {benchmark_annual_ret * 100:.2f}%")
col4.metric("平均換手率", f"{avg_turnover * 100:.2f}%")
col5.metric("多空預測勝率", f"{win_rate:.2f}%", f"正確數: {correct_count}/{total_count}")

st.markdown("---")

st.subheader("📈 累積淨值曲線對比")
comparison_df = pd.DataFrame({
    f"動態配置策略 ({chosen_defense_mode})": cum_returns,
    "S&P 500 (買入持有)": benchmark_cum
})
fig_wealth = px.line(comparison_df, labels={"value": "累積淨值", "index": "日期", "variable": "策略類型"})
fig_wealth.update_layout(height=450, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_wealth, use_container_width=True)

st.subheader("🌍 全球市場多空狀態歷史判定圖")
fig_regime = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("S&P 500 走勢", "模型預測牛市機率 (Prob Bull)")
)

sp_prices_plot = (1 + benchmark_returns).cumprod()
fig_regime.add_trace(
    go.Scatter(x=sp_prices_plot.index, y=sp_prices_plot, name="S&P 500 走勢", line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

bull_probs = res_df['Global_Bull_Prob']
fig_regime.add_trace(
    go.Scatter(x=bull_probs.index, y=bull_probs, name="牛市機率", line=dict(color='#ff7f0e', width=1.5)),
    row=2, col=1
)
fig_regime.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1, annotation_text="多空分界線 (0.5)")

fig_regime.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_regime, use_container_width=True)

st.markdown("---")

st.subheader(f"📋 週期結算、預測與未來報酬驗證明細表 ({rebal_freq_option})")
st.caption("明細表中已包含最新收盤日的訊號（最上方會顯示『待下期結算 / 待驗證』）：")
st.dataframe(pd.DataFrame(period_records), use_container_width=True, height=400)
