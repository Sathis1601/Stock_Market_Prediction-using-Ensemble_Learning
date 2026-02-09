"""
Data Collection Module
Fetches historical stock market data using yfinance API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
from pathlib import Path


class StockDataCollector:
    """
    Collects and manages stock market data from yfinance
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize data collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def fetch_stock_data(self, ticker, start_date=None, end_date=None, save=True):
        """
        Fetch historical stock data for a given ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', '^GSPC')
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        save : bool
            Whether to save the data to disk
            
        Returns:
        --------
        pd.DataFrame
            Stock data with OHLCV columns
        """
        if start_date is None:
            start_date = self.data_config['start_date']
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        
        try:
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Keep only OHLCV columns
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date'}, inplace=True)
            
            print(f"✓ Successfully fetched {len(df)} records for {ticker}")
            
            if save:
                self._save_raw_data(df, ticker)
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data for {ticker}: {str(e)}")
            raise
    
    def _save_raw_data(self, df, ticker):
        """Save raw data to CSV"""
        filename = f"{ticker.replace('^', '').replace('.', '_')}_raw.csv"
        filepath = self.raw_data_path / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved raw data to {filepath}")
    
    def load_raw_data(self, ticker):
        """Load previously saved raw data"""
        filename = f"{ticker.replace('^', '').replace('.', '_')}_raw.csv"
        filepath = self.raw_data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"No saved data found for {ticker}")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def fetch_multiple_tickers(self, tickers, start_date=None, end_date=None):
        """
        Fetch data for multiple tickers
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
            
        Returns:
        --------
        dict
            Dictionary mapping ticker to DataFrame
        """
        data_dict = {}
        
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, start_date, end_date)
                data_dict[ticker] = df
            except Exception as e:
                print(f"Skipping {ticker} due to error: {str(e)}")
        
        return data_dict
    
    def get_latest_data(self, ticker, days=365):
        """
        Get the most recent data for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days : int
            Number of days of historical data to fetch
            
        Returns:
        --------
        pd.DataFrame
            Recent stock data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_stock_data(
            ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            save=False
        )
    
    def validate_data(self, df):
        """
        Validate the fetched data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data to validate
            
        Returns:
        --------
        dict
            Validation results
        """
        validation = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': (df['date'].min(), df['date'].max()),
            'price_range': (df['close'].min(), df['close'].max()),
            'volume_range': (df['volume'].min(), df['volume'].max()),
        }
        
        # Check for anomalies
        validation['anomalies'] = []
        
        # Check for zero or negative prices
        if (df['close'] <= 0).any():
            validation['anomalies'].append('Zero or negative prices found')
        
        # Check for zero volume
        if (df['volume'] == 0).sum() > len(df) * 0.05:  # More than 5%
            validation['anomalies'].append('High percentage of zero volume days')
        
        # Check for missing dates (gaps)
        date_diff = df['date'].diff()
        large_gaps = date_diff[date_diff > timedelta(days=7)]
        if len(large_gaps) > 0:
            validation['anomalies'].append(f'{len(large_gaps)} date gaps > 7 days')
        
        return validation


def main():
    """Example usage"""
    # Initialize collector
    collector = StockDataCollector()
    
    # Fetch data for default tickers
    default_tickers = [
        '^GSPC',  # S&P 500
        'AAPL',   # Apple
        'GOOGL',  # Google
        'MSFT',   # Microsoft
    ]
    
    print("=" * 60)
    print("Stock Market Data Collection")
    print("=" * 60)
    
    for ticker in default_tickers:
        try:
            df = collector.fetch_stock_data(ticker)
            validation = collector.validate_data(df)
            
            print(f"\n{ticker} Validation:")
            print(f"  Records: {validation['total_records']}")
            print(f"  Date Range: {validation['date_range'][0]} to {validation['date_range'][1]}")
            print(f"  Price Range: ${validation['price_range'][0]:.2f} - ${validation['price_range'][1]:.2f}")
            
            if validation['anomalies']:
                print(f"  ⚠ Anomalies: {', '.join(validation['anomalies'])}")
            else:
                print(f"  ✓ Data quality looks good")
            
        except Exception as e:
            print(f"✗ Failed to fetch {ticker}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
