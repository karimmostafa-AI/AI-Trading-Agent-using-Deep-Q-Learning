import yfinance as yf
import pandas as pd
import numpy as np
import time

def load_stock_data(symbol, start_date, end_date, max_retries=3):
    """
    Load and preprocess stock market data with error handling and retries.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_retries (int): Maximum number of download attempts
        
    Returns:
        pd.DataFrame: Preprocessed stock data
    """
    for attempt in range(max_retries):
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                raise ValueError(f"No data downloaded for {symbol}")
            
            # Verify we have the essential columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns for {symbol}")
            
            # Calculate technical indicators
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Returns'] = data['Close'].pct_change()
            
            # Add more technical indicators
            data['RSI'] = calculate_rsi(data['Close'])
            data['MACD'], data['Signal'] = calculate_macd(data['Close'])
            
            # Drop NaN values and reset index
            data.dropna(inplace=True)
            
            if len(data) < 30:  # Ensure we have enough data points
                raise ValueError(f"Insufficient data points for {symbol}")
                
            data.reset_index(inplace=True)
            return data
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise ValueError(f"Failed to download data for {symbol} after {max_retries} attempts")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def normalize_data(data):
    """
    Normalize the data using min-max scaling.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Normalized data
    """
    result = data.copy()
    for column in result.columns:
        if column != 'Date':
            min_val = result[column].min()
            max_val = result[column].max()
            result[column] = (result[column] - min_val) / (max_val - min_val)
    return result 