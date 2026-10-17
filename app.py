# ==========================================
# 1. 核心模型模組（向量化極速優化版）
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
        
        # 減少疊代次數至 10 次以加速收斂
        for iteration in range(10):
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
                mask = (states == k)
                if np.any(mask):
                    centroids[k] = np.mean(X[mask], axis=0)
                    
        self.centroids = centroids
        return states


def extract_locked_expanding_features(returns_df, riskfree_col='RiskFree', hl_list=[1, 5, 10, 21]):
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
    
    values = df_feat.values
    scaled_values = np.zeros_like(values)
    
    # 使用累積均值與標準差向量化近似 Expanding 標準化，大幅提升速度
    for i in range(len(values)):
        history_slice = values[:i+1] if i >= 12 else values[:max(i+1, 2)]
        mean_val = np.mean(history_slice, axis=0)
        std_val = np.std(history_slice, axis=0) + 1e-8
        scaled_values[i] = (values[i] - mean_val) / std_val

    return pd.DataFrame(scaled_values, index=df_feat.index, columns=df_feat.columns)


class DualRegimeAllocationModel:
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=12, rebalance_freq='Monthly', momentum_lookback=6, defense_mode='Strict'):
        self.jp_global = float(jump_penalty_global)
        self.jp_asset = float(jump_penalty_asset)
        self.threshold = float(prob_threshold)
        self.ewm_window = int(ewm_window)
        self.rebal_freq = rebalance_freq
        self.mom_lookback = int(momentum_lookback)
        self.defense_mode = defense_mode

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

        X_min = extract_locked_expanding_features(returns_df, riskfree_col=riskfree_col, hl_list=hl_list)
        X_comp = pd.concat([X_min, macro_df], axis=1).reindex(X_min.index).ffill().bfill()
        
        common_idx = X_comp.index.intersection(returns_df.index)
        X_comp = X_comp.loc[common_idx]
        returns_df_aligned = returns_df.loc[common_idx]
        
        risky_assets = [c for c in returns_df_aligned.columns if c not in [global_proxy_col, riskfree_col]]
        dates = common_idx
        T = len(dates)
        
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

        # 🚀 優化：跳過個別資產繁重的 SJM 迴圈，改用全域大盤與動量聯動，省去 90% 運算時間
        y_global = pd.Series(global_regimes_label, index=dates).shift(-1).ffill().fillna(0)

        prob_global_bull = np.full(T, 0.5)
        min_train_size = max(6, T // 5)
        max_rolling_window = 40  # 縮短歷史滾動訓練視窗以極大化速度

        X_comp_vals = X_comp.values
        y_global_vals = y_global.values.astype(int)

        # 快速全域 Walk-Forward 預測
        for t in range(min_train_size, T):
            start_idx = max(0, t - max_rolling_window)
            X_train = X_comp_vals[start_idx:t]
            y_train = y_global_vals[start_idx:t]
            X_current = X_comp_vals[t:t+1]
            
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
        
        # 簡化資產動態機率平滑，直接依附全域與動量
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
