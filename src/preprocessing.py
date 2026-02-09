"""
Preprocessing and Feature Engineering Module
Calculates technical indicators and prepares data for deep learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
from pathlib import Path
import joblib


class FeatureEngineer:
    """
    Feature engineering for stock market data
    Calculates technical indicators and creates sequences for time-series models
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.lookback = self.feature_config['lookback_window']
        self.prediction_horizon = self.feature_config['prediction_horizon']
        self.volatility_window = self.feature_config['volatility_window']
        
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = MinMaxScaler()
    
    def calculate_log_returns(self, df):
        """Calculate logarithmic returns"""
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def calculate_rolling_volatility(self, df, window=None):
        """
        Calculate rolling volatility (standard deviation of returns)
        This is our TARGET variable
        """
        if window is None:
            window = self.volatility_window
        
        # First calculate returns if not already done
        if 'log_returns' not in df.columns:
            df = self.calculate_log_returns(df)
        
        # Calculate rolling volatility
        df['volatility'] = df['log_returns'].rolling(window=window).std()
        
        # Annualized volatility (multiply by sqrt(252) for daily data)
        df['volatility_annualized'] = df['volatility'] * np.sqrt(252)
        
        return df
    
    def calculate_sma(self, df, window):
        """Calculate Simple Moving Average"""
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        return df
    
    def calculate_ema(self, df, span):
        """Calculate Exponential Moving Average"""
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        return df
    
    def calculate_rsi(self, df, window=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero and handle inf values
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Replace inf and -inf with NaN
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        df[f'rsi_{window}'] = rsi
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        return df
    
    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        
        df['bb_upper'] = sma + (std * num_std)
        df['bb_lower'] = sma - (std * num_std)
        df['bb_middle'] = sma
        # Avoid division by zero
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        df['bb_width'] = df['bb_width'].replace([np.inf, -np.inf], np.nan)
        return df
    
    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df[f'atr_{window}'] = true_range.rolling(window).mean()
        return df
    
    def calculate_momentum(self, df, window=10):
        """Calculate price momentum"""
        df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
        return df
    
    def calculate_all_features(self, df):
        """
        Calculate all technical indicators based on configuration
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw stock data with OHLCV columns
            
        Returns:
        --------
        pd.DataFrame
            Data with all calculated features
        """
        print("Calculating technical indicators...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate all indicators
        df = self.calculate_log_returns(df)
        df = self.calculate_rolling_volatility(df)
        df = self.calculate_sma(df, 20)
        df = self.calculate_sma(df, 50)
        df = self.calculate_ema(df, 12)
        df = self.calculate_ema(df, 26)
        df = self.calculate_rsi(df, 14)
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_atr(df, 14)
        df = self.calculate_momentum(df, 10)
        
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        
        # Clean up any inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print(f"✓ Calculated {len(df.columns) - 7} technical indicators")
        
        return df
    
    def prepare_target_variable(self, df):
        """
        Prepare target variable (future volatility)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with calculated features
            
        Returns:
        --------
        pd.DataFrame
            Data with target variable
        """
        # Target: volatility N days ahead
        df['target_volatility'] = df['volatility'].shift(-self.prediction_horizon)
        
        return df
    
    def create_sequences(self, df, feature_columns):
        """
        Create sequences for time-series deep learning models
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with all features
        feature_columns : list
            List of column names to use as features
            
        Returns:
        --------
        tuple
            (X, y) where X is (samples, timesteps, features) and y is (samples,)
        """
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Extract features and target
        features = df_clean[feature_columns].values
        target = df_clean['target_volatility'].values
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(features) - self.lookback):
            X.append(features[i:i + self.lookback])
            y.append(target[i + self.lookback - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Created {len(X)} sequences with shape: {X.shape}")
        
        return X, y
    
    def normalize_features(self, X_train, X_val=None, X_test=None):
        """
        Normalize features using MinMaxScaler
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training data (samples, timesteps, features)
        X_val : np.ndarray, optional
            Validation data
        X_test : np.ndarray, optional
            Test data
            
        Returns:
        --------
        tuple
            Normalized arrays
        """
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        results = [X_train_scaled]
        
        # Transform validation and test data if provided
        if X_val is not None:
            n_samples_val = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
            results.append(X_val_scaled)
        
        if X_test is not None:
            n_samples_test = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def save_scaler(self, filepath):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
        print(f"✓ Saved scaler to {filepath}")
    
    def load_scaler(self, filepath):
        """Load a previously fitted scaler"""
        self.scaler = joblib.load(filepath)
        print(f"✓ Loaded scaler from {filepath}")
    
    def get_feature_columns(self):
        """
        Get list of feature columns to use for model training
        
        Returns:
        --------
        list
            List of feature column names
        """
        # Core price features
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'log_returns', 'volatility',
            'price_change', 'volume_change', 'hl_spread'
        ]
        
        # Moving averages
        features.extend(['sma_20', 'sma_50', 'ema_12', 'ema_26'])
        
        # Momentum indicators
        features.extend(['rsi_14', 'macd', 'macd_signal', 'macd_diff'])
        
        # Bollinger Bands
        features.extend(['bb_upper', 'bb_lower', 'bb_middle', 'bb_width'])
        
        # Volatility indicators
        features.extend(['atr_14', 'momentum_10'])
        
        return features
    
    def process_and_save(self, df, ticker):
        """
        Process data and save to disk
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw stock data
        ticker : str
            Stock ticker symbol
        """
        # Calculate all features
        df_processed = self.calculate_all_features(df)
        df_processed = self.prepare_target_variable(df_processed)
        
        # Save processed data
        filename = f"{ticker.replace('^', '').replace('.', '_')}_processed.csv"
        filepath = self.processed_data_path / filename
        df_processed.to_csv(filepath, index=False)
        
        print(f"✓ Saved processed data to {filepath}")
        
        return df_processed


def main():
    """Example usage"""
    from data_collection import StockDataCollector
    
    print("=" * 60)
    print("Feature Engineering Demo")
    print("=" * 60)
    
    # Collect data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('^GSPC', save=False)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Calculate features
    df_processed = engineer.calculate_all_features(df)
    df_processed = engineer.prepare_target_variable(df_processed)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Features: {list(df_processed.columns)}")
    
    # Create sequences
    feature_columns = engineer.get_feature_columns()
    X, y = engineer.create_sequences(df_processed, feature_columns)
    
    print(f"\nSequence shape: X={X.shape}, y={y.shape}")
    print(f"Target volatility range: {y.min():.6f} - {y.max():.6f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
