import pandas as pd
import numpy as np
import pyodbc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import ta
import warnings
warnings.filterwarnings('ignore')

class XGBoostSwingTrader:
    def __init__(self, server, database, table_name='daily_prices', 
                 start_date='2015-01-01', end_date='2025-12-05'):
        self.server = server
        self.database = database
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        self.model_5d = None
        self.model_10d = None
        
    def fetch_data(self):
        """Fetch OHLCV data from SQL Server table"""
        print("Fetching OHLCV data from SQL Server...")
        
        # SQL Server connection string (update with your credentials)
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Trusted_Connection=yes;"
        )
        
        query = f"""
        SELECT Symbol, Date, Open, High, Low, Close, Volume
        FROM {self.table_name}
        WHERE Date >= '{self.start_date}' AND Date <= '{self.end_date}'
        ORDER BY Symbol, Date
        """
        
        try:
            self.data = pd.read_sql(query, pyodbc.connect(conn_str))
            print(f"Fetched {len(self.data)} rows from SQL Server")
            print(f"Symbols: {self.data['Symbol'].nunique()}")
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            
            # Quick data quality check
            print("Data summary:")
            print(self.data.groupby('Symbol').size().describe())
            
        except Exception as e:
            print(f"SQL fetch error: {e}")
            raise
    
    def create_labels(self, horizon=5):
        """Create 5/10-day forward return labels"""
        self.data['fwd_return'] = self.data.groupby('Symbol')['Close'].pct_change(horizon).shift(-horizon)
        self.data['label'] = np.where(self.data['fwd_return'] > 0.03, 1, 
                                     np.where(self.data['fwd_return'] < -0.03, -1, 0))
        return self.data
    
    def engineer_features(self):
        """Generate ~400 OHLCV + technical features"""
        print("Engineering features...")
        df = self.data.copy()
        
        # Price/Returns (50+ features)
        for days in [1,2,3,5,10,20,40]:
            df[f'ret_{days}d'] = df.groupby('Symbol')['Close'].pct_change(days)
            df[f'high_{days}d'] = df.groupby('Symbol')['High'].rolling(days).max().shift(1)
            df[f'low_{days}d'] = df.groupby('Symbol')['Low'].rolling(days).min().shift(1)
            df[f'range_{days}d'] = (df[f'high_{days}d'] - df[f'low_{days}d']) / df['Close']
        
        # Volume features (50+)
        for days in [5,10,20]:
            df[f'vol_ma_{days}'] = df.groupby('Symbol')['Volume'].rolling(days).mean().shift(1)
            df[f'rel_vol_{days}'] = df['Volume'] / df[f'vol_ma_{days}']
            df[f'dollar_vol_{days}'] = (df['Close'] * df['Volume']).rolling(days).mean().shift(1)
        
        # Volume-Price interactions (high importance)
        df['vol_price'] = df['Volume'] * df['ret_1d']
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        for days in [5,10]:
            df[f'obv_slope_{days}'] = df.groupby('Symbol')['obv'].diff(days) / days
        
        # Volatility (40+)
        for days in [5,10,20]:
            df[f'ret_vol_{days}'] = df.groupby('Symbol')['ret_1d'].rolling(days).std().shift(1)
            df[f'atr_{days}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], days)
            df[f'atr_pct_{days}'] = df[f'atr_{days}'] / df['Close']
        
        # Momentum indicators (50+)
        rsi_periods = [5,10,14]
        for p in rsi_periods:
            df[f'RSI_{p}'] = ta.momentum.rsi(df['Close'], window=p)
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Moving averages (50+)
        ma_periods = [5,10,20,50]
        for p in ma_periods:
            df[f'SMA_{p}'] = ta.trend.sma_indicator(df['Close'], window=p)
            df[f'price_ma_{p}'] = (df['Close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']
            df[f'SMA_slope_{p}'] = df[f'SMA_{p}'].pct_change(5)
        
        # Candle patterns (20+)
        df['body_size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-8)
        df['upper_wick'] = (df['High'] - df[['Open','Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-8)
        df['lower_wick'] = (df[['Open','Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        # Cross-sectional ranks (daily) - Top momentum/volume features
        momentum_cols = [f'ret_{d}d' for d in [1,2,3,5]] + ['RSI_14', 'MACD_hist', 'obv_slope_5']
        volume_cols = ['rel_vol_5', 'rel_vol_10', 'vol_price']
        
        rank_cols = momentum_cols + volume_cols
        for col in rank_cols:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('Date')[col].rank(pct=True)
        
        self.features = [col for col in df.columns if col not in 
                        ['Date','Open','High','Low','Close','Volume','Symbol','fwd_return','label']]
        print(f"Generated {len(self.features)} features")
        return df[self.features + ['label', 'Symbol', 'Date', 'fwd_return']]
    
    def prepare_data(self):
        """Clean and prepare training data"""
        self.create_labels(horizon=5)  # 5-day first
        df = self.engineer_features()
        
        # Remove NaNs, filter liquid stocks (top 90% avg volume)
        df = df.dropna(subset=['label'] + self.features)
        vol_threshold = df.groupby('Symbol')['Volume'].quantile(0.1)
        df = df.merge(vol_threshold.rename('vol_min'), on='Symbol', how='left')
        df = df[df['Volume'] > df['vol_min']].drop('vol_min', axis=1)
        
        print(f"Prepared data shape: {df.shape}")
        return df
    
    def train_models(self, df):
        """Train separate 5-day and 10-day XGBoost models"""
        print("Training XGBoost models...")
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        X = df[self.features]
        y = df['label']
        
        # 5-day model
        self.model_5d = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            scale_pos_weight=2  # Handle class imbalance
        )
        
        # Train with time-series CV
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model_5d.fit(X_train, y_train)
            preds = self.model_5d.predict(X_val)
            score = accuracy_score(y_val, preds)
            cv_scores.append(score)
            print(f"5D CV fold accuracy: {score:.3f}")
        
        print(f"5D CV Mean accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")
        
        # Train final 5D model on full data
        self.model_5d.fit(X, y)
        
        # 10-day model
        df_10d = df.copy()
        df_10d['fwd_return'] = df_10d.groupby('Symbol')['Close'].pct_change(10).shift(-10)
        df_10d['label'] = np.where(df_10d['fwd_return'] > 0.03, 1, 
                                  np.where(df_10d['fwd_return'] < -0.03, -1, 0))
        df_10d = df_10d.dropna(subset=['label'])
        
        X_10d = df_10d[self.features]
        y_10d = df_10d['label']
        
        self.model_10d = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            scale_pos_weight=2
        )
        self.model_10d.fit(X_10d, y_10d)
        
        # Feature importance (5D model)
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model_5d.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        print("\nTop 20 Features (5D Model):")
        print(importance_df)
        
        return self.model_5d, self.model_10d
    
    def generate_signals(self, latest_data):
        """Generate live trading signals"""
        if len(latest_data) == 0:
            return None
            
        X_live = latest_data[self.features].iloc[-1:]  # Latest row
        pred_5d = self.model_5d.predict_proba(X_live)[0]
        pred_10d = self.model_10d.predict_proba(X_live)[0]
        
        signal = 'HOLD'
        long_prob = pred_5d[2]  # Class 1 (long)
        short_prob = pred_5d[0] # Class -1 (short)
        
        if long_prob > 0.55:
            signal = 'LONG'
        elif short_prob > 0.55:
            signal = 'SHORT'
            
        return {
            'symbol': latest_data['Symbol'].iloc[-1],
            'date': latest_data['Date'].iloc[-1],
            'signal': signal,
            'prob_long_5d': long_prob,
            'prob_short_5d': short_prob,
            'prob_long_10d': pred_10d[2],
            'prob_short_10d': pred_10d[0]
        }

# Usage Example
if __name__ == "__main__":
    # Update these with your SQL Server details
    trader = XGBoostSwingTrader(
        server='your_server_name',  # e.g., 'localhost' or 'server.domain.com'
        database='your_database_name',  # e.g., 'TradingData'
        table_name='daily_prices'  # Your OHLCV table
    )
    
    # Full pipeline
    trader.fetch_data()
    df_prepared = trader.prepare_data()
    trader.train_models(df_prepared)
    
    # Generate signals for latest data per symbol
    latest_signals = []
    for symbol in df_prepared['Symbol'].unique():
        latest = df_prepared[df_prepared['Symbol'] == symbol].tail(1)
        signal = trader.generate_signals(latest)
        if signal:
            latest_signals.append(signal)
    
    signals_df = pd.DataFrame(latest_signals)
    print("\nLive Trading Signals:")
    print(signals_df.sort_values('prob_long_5d', ascending=False))
    
    # Save signals back to SQL Server
    # signals_df.to_sql('daily_signals', trader.engine, if_exists='append', index=False)
