import gym
import numpy as np
import pandas as pd
import pandas_ta as ta
import os
import glob
from gym import spaces

class CryptoPortfolioEnv(gym.Env):
    def __init__(self, data_dir='data/1h', top_k=4, initial_balance=10000.0, fee_rate=0.001):
        super(CryptoPortfolioEnv, self).__init__()
        
        self.data_dir = data_dir
        self.top_k = top_k
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        
        # 1. Load Data
        # self.market_data shape: [Time, Num_Coins, Num_Features]
        # self.actual_returns shape: [Time, Num_Coins] (For Reward Calculation)
        self.assets, self.dates, self.market_data, self.actual_returns = self._load_and_process_data()
        self.n_assets = len(self.assets)
        self.n_features = self.market_data.shape[2]
        self.max_steps = len(self.dates) - 1
        
        print(f"Env Initialized: {self.n_assets} assets, {self.max_steps} steps, {self.n_features} features.")
        
        # 2. Define Action Space
        # [Score_Asset1, ..., Score_AssetN, Score_Cash]
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets + 1,), dtype=np.float32
        )
        
        # 3. Define Observation Space
        # Market Data (N*F) + Portfolio Weights (N+1)
        obs_dim = (self.n_assets * self.n_features) + (self.n_assets + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal State
        self.current_step = 0
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[-1] = 1.0 # Start with 100% Cash
        self.portfolio_value = self.initial_balance

    def _load_and_process_data(self):
        """
        Loads parquet files, calculates Hybrid Features (Z-Score + Abs).
        Returns: assets, common_index, market_data_tensor, actual_returns_matrix
        """
        file_pattern = os.path.join(self.data_dir, "*_3y.parquet")
        files = glob.glob(file_pattern)
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
            
        data_map = {}
        assets = []
        btc_close = None
        
        print(f"Loading {len(files)} files/assets...")
        
        # 1. Initial Load & Identify BTC
        temp_dfs = {}
        for f in files:
            asset_name = os.path.basename(f).split('_')[0].replace('USDT', '') # BTCUSDT -> BTC
            df = pd.read_parquet(f)
            df = df.sort_values('datetime').set_index('datetime')
            
            # Basic validation
            if 'close' not in df.columns: continue
            
            temp_dfs[asset_name] = df
            assets.append(asset_name)
            
            if asset_name == 'BTC':
                btc_close = df['close']
        
        assets.sort()
        
        # 2. Find Common Index (Intersection)
        if not assets: raise ValueError("No assets loaded.")
        common_index = temp_dfs[assets[0]].index
        for asset in assets[1:]:
            common_index = common_index.intersection(temp_dfs[asset].index)
            
        if len(common_index) == 0: raise ValueError("No overlapping dates found!")
        
        # Filter BTC to common index for correlation
        if btc_close is not None:
            btc_close = btc_close.reindex(common_index).fillna(method='ffill')
            btc_ret = np.log(btc_close / btc_close.shift(1))
        
        # 3. Calculate Features per Asset
        processed_data_list = []
        
        # Window sizes
        if '1h' in self.data_dir:
            ROLL_WIN = 24  # 24h
            # Normalized Volume (simple z-scoreish or relative)
            # Use bfill() directly
            # This line is added as per the instruction, assuming it's a new feature.
            # It's placed here because the instruction snippet shows it under the '1h' condition.
            # df['vol_norm'] = (df['volume'] / df['volume'].rolling(24).mean().bfill()) - 1
        else:
            ROLL_WIN = 96  # 24h (15m * 4 * 24)
            
        print(f"Processing features with Rolling Window = {ROLL_WIN}...")
        
        final_feature_list = [] # For return tensor
        actual_returns_list = [] # For reward
        
        for asset in assets:
            df = temp_dfs[asset].reindex(common_index).ffill() # Align
            
            # --- A. Raw Calculations ---
            # 1. Log Return (Actual)
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
            # 2. Log Volume
            df['log_vol'] = np.log(df['volume'] + 1.0)
            
            # --- B. Hybrid Features ---
            
            # 1. Rolling Z-Score Return (Pattern)
            # z_score = (x - mean) / std
            roll_mean_ret = df['log_ret'].rolling(ROLL_WIN).mean()
            roll_std_ret = df['log_ret'].rolling(ROLL_WIN).std()
            df['z_ret'] = (df['log_ret'] - roll_mean_ret) / (roll_std_ret + 1e-8)
            
            # 2. Rolling Z-Score Volume (Liquidity Pattern)
            roll_mean_vol = df['log_vol'].rolling(ROLL_WIN).mean()
            roll_std_vol = df['log_vol'].rolling(ROLL_WIN).std()
            df['z_vol'] = (df['log_vol'] - roll_mean_vol) / (roll_std_vol + 1e-8)
            
            # 3. RSI (Bounded Pattern)
            df['rsi'] = df.ta.rsi(length=14) / 100.0
            
            # 4. PPO (Trend Pattern)
            # PPO is percentage: (FastEMA - SlowEMA) / SlowEMA. Usually 1.0 = 1%. 
            # We divide by 100 to make it 0.01 scale like returns if needed, or keep as is.
            # PPO output is a DF with [PPO, Signal, Hist]. We take PPO buffer.
            ppo_df = df.ta.ppo(fast=12, slow=26, signal=9)
            if ppo_df is not None:
                df['ppo'] = ppo_df.iloc[:, 0] / 100.0 # PPO_12_26_9
            else:
                df['ppo'] = 0.0
                
            # 5. ATR% (Absolute Volatility Magnitude)
            # ATR is absolute value ($). Divide by Close to get %.
            df['atr'] = df.ta.atr(length=14)
            df['atr_pct'] = df['atr'] / df['close']
            
            # 6. BTC Correlation (Regime Context)
            if btc_close is not None:
                # Rolling correlation of returns
                df['btc_corr'] = df['log_ret'].rolling(ROLL_WIN).corr(btc_ret)
            else:
                df['btc_corr'] = 0.0 # Should not happen if BTC is in list
                
            # --- Clean up ---
            # Drop NaN rows generated by rolling/shift
            # (We will handle this by dropping head of ALL assets later)
            
            # Select Final Features
            feats = ['z_ret', 'z_vol', 'rsi', 'ppo', 'atr_pct', 'btc_corr']
            
            # Store
            final_feature_list.append(df[feats])
            actual_returns_list.append(df['log_ret'])
            
        # 4. Align & Drop NaNs (Global)
        # Find max NaN index across all assets (usually ROLL_WIN + some indicator warmup)
        # RSI needs 14, PPO needs 26, Rolling needs ROLL_WIN. Max is ~ROLL_WIN.
        
        # Concatenate temporarily to find valid start
        # Use first asset to check index validity
        valid_idx_start = 0
        test_df = final_feature_list[0]
        # Find first index where no feature is NaN
        for i in range(len(test_df)):
            if not test_df.iloc[i].isnull().any():
                valid_idx_start = i
                break
        
        # Add a safety buffer
        valid_idx_start += 1
        
        # Apply slice
        final_common_index = common_index[valid_idx_start:]
        
        print(f"Data Processed. Trimmed initial {valid_idx_start} steps for warmup.")
        
        # Convert to Tensor
        market_data_np = [] # [Time, Asset, Feature]
        actual_ret_np = []  # [Time, Asset]
        
        for i, asset in enumerate(assets):
            # Features
            f_df = final_feature_list[i].iloc[valid_idx_start:]
            market_data_np.append(f_df.values)
            
            # Returns
            r_series = actual_returns_list[i].iloc[valid_idx_start:]
            actual_ret_np.append(r_series.values)
            
        # Transpose to [Time, Asset, Feature] from [Asset, Time, Feature]
        market_data_np = np.array(market_data_np, dtype=np.float32).transpose(1, 0, 2)
        actual_ret_np = np.array(actual_ret_np, dtype=np.float32).transpose(1, 0)
        
        return assets, final_common_index, market_data_np, actual_ret_np

    def seed(self, seed=None):
        """
        Gym compatibility: Set random seed.
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def get_task(self):
        """
        Dummy task interface for variBAD compatibility.
        """
        return None
        
    def reset_task(self, task=None):
        """
        Dummy task interface for variBAD compatibility.
        """
        # We handle task variability via random reset()
        pass

    def reset(self):
        """
        Resets environment to a random start point.
        """
        # Random start, ensuring we have enough data (already trimmed, but need space for episode)
        # Episode length: let's say 2000 steps (matching __init__ registration)
        # We just pick a random start between [0, max_steps - window]
        # Safety check: if max_steps < 2000, we start at 0.
        
        limit = max(1, self.max_steps - 2000)
        self.current_step = np.random.randint(0, limit)
        
        # Reset Portfolio
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[-1] = 1.0 # 100% Cash
        self.portfolio_value = self.initial_balance
        
        return self._get_obs()
        
    def step(self, action):
        """
        1. Parse Action (Logits -> Top-K Signed Weights)
        2. Calculate Returns & Costs
        3. Update Portfolio
        4. Return Obs, Reward, Done
        """
        # --- A. Action Parsing (Long/Short + Cash) ---
        # action: [Score_Asset1, ..., Score_Asset10, Score_Cash]
        
        # 1. Separate Cash Score
        cash_score = action[-1]
        asset_scores = action[:-1]
        
        # 2. Identify Top-K using Absolute Confidence
        # We want magnitude of conviction, direction comes from sign
        abs_scores = np.abs(asset_scores)
        
        # Indicies of top-k assets
        top_k_indices = np.argsort(abs_scores)[-self.top_k:]
        
        # 3. Create Masked Score Vector for Softmax
        # We need positive scores for Softmax to determine magnitude
        # We include Cash in this competition
        relevant_scores = np.full(self.n_assets + 1, -np.inf) # Start with -inf (0 prob)
        
        # Fill in Top-K asset scores (using ABSOLUTE value for sizing)
        relevant_scores[top_k_indices] = abs_scores[top_k_indices]
        # Fill in Cash score (Cash is always 'Long' magnitude, 0~1)
        relevant_scores[-1] = cash_score
        
        # 4. Apply Softmax to determine Size (Magnitude)
        # Shift for stability
        exp_scores = np.exp(relevant_scores - np.max(relevant_scores))
        magnitude_weights = exp_scores / np.sum(exp_scores)
        
        # 5. Apply Signs (Restore Directions)
        # Cash is always positive (index -1)
        target_weights = np.zeros(self.n_assets + 1)
        
        # Assets: Weight = Magnitude * Sign(OriginalScore)
        # If score was 0, sign is 0.
        target_weights[:-1] = magnitude_weights[:-1] * np.sign(asset_scores)
        # Cash
        target_weights[-1] = magnitude_weights[-1]
        
        # --- B. Calculate Portfolio Return ---
        # Get Returns for 'current_step' (Reaction to previous step's action? No, RL usually:
        # Obs_t -> Action_t -> Step -> Reward_{t+1} (Price_t to Price_{t+1})
        
        # Market Returns at t -> t+1
        # self.actual_returns shape: [Time, Assets]
        # current_step is t. We need return from t to t+1. 
        # So we look at actual_returns[current_step + 1] (Future Return)
        
        asset_returns = self.actual_returns[self.current_step + 1] # [N_Assets]
        
        # Weighted Return calculation
        # If Long (W>0) and Ret>0 -> +, Ret<0 -> -
        # If Short (W<0) and Ret>0 -> - (W*R is neg), Ret<0 -> + (Neg*Neg is Pos)
        # Perfect. Simple dot product works for Long/Short PnL.
        
        # Cash return is 0 (or risk free rate, current 0)
        gross_return = np.sum(target_weights[:-1] * asset_returns)
        
        # --- C. Transaction Costs ---
        # Diff from PREVIOUS weights (self.weights) to TARGET weights
        # We assume rebalancing happens instantly at Open of t
        turnover = np.sum(np.abs(target_weights - self.weights))
        cost = turnover * self.fee_rate
        
        net_log_return = gross_return - cost
        
        # --- D. Update Internal State ---
        self.weights = target_weights
        self.portfolio_value *= np.exp(net_log_return)
        
        # --- E. Calculate Reward (Risk Adjusted) ---
        # Downside Penalty (Loss Aversion)
        # If purely losing money, multiply return by 2.0 (make it more negative)
        if gross_return < 0:
            risk_adjusted_return = gross_return * 2.0
        else:
            risk_adjusted_return = gross_return
            
        reward = risk_adjusted_return - cost # Cost is always painful
        
        # Optional: Clip reward for stability?
        # reward = np.clip(reward, -1.0, 1.0) # Maybe later
        
        # --- F. Next Step & Done ---
        self.current_step += 1
        done = (self.current_step >= self.max_steps - 1)
        
        # Check Bankruptcy (Value < 10% of initial) -> Force Done
        if self.portfolio_value < self.initial_balance * 0.1:
            done = True
            reward = -10.0 # Heavy penalty for death
        
        obs = self._get_obs()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': net_log_return,
            'cost': cost,
            'turnover': turnover
        }
        
        return obs, reward, done, info
        
    def _get_obs(self):
        # 1. Market Data [Assets, Features]
        # Flattened
        market_obs = self.market_data[self.current_step].flatten()
        
        # 2. Portfolio State [Weights]
        portfolio_obs = self.weights
        
        return np.concatenate([market_obs, portfolio_obs])
