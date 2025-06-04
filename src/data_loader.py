from typing import Dict, Optional, List, Union
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path

class DataLoaderError(Exception):
    """Custom exception for data loading errors"""
    pass

class StockDataLoader:
    """Robust stock data loader with error handling"""
    
    def __init__(self, cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def load_stock_data(self, 
                       symbols: List[str], 
                       start_date: Union[str, datetime], 
                       end_date: Union[str, datetime],
                       use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Load stock data with caching and error handling"""
        
        stock_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"Loading data for {symbol}")
                
                # Check cache first
                cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.csv"
                
                if use_cache and cache_file.exists():
                    self.logger.info(f"Loading {symbol} from cache")
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                else:
                    # Download from yfinance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if df.empty:
                        raise DataLoaderError(f"No data found for {symbol}")
                    
                    # Cache the data
                    if use_cache:
                        df.to_csv(cache_file)
                        self.logger.info(f"Cached data for {symbol}")
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                stock_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                
        if failed_symbols:
            self.logger.warning(f"Failed to load data for: {failed_symbols}")
            
        if not stock_data:
            raise DataLoaderError("No stock data could be loaded")
            
        return stock_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to stock data"""
        try:
            # Daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Volatility
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # RSI (simplified)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df