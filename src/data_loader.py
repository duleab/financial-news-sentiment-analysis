# Data Loader and Preprocessing Script
# For Financial News Sentiment Analysis Project

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """
    Data loader class for financial news and stock price data
    """
    
    def __init__(self, data_directory="data"):
        """
        Initialize the data loader
        
        Args:
            data_directory (str): Path to the data directory
        """
        self.data_dir = Path(data_directory)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.yfinance_data_dir = self.data_dir / "yfinance_data"
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_news_data(self, filename=None):
        """
        Load financial news data
        
        Args:
            filename (str): Specific filename to load, if None will look for common patterns
            
        Returns:
            pd.DataFrame: Loaded news data
        """
        print("Loading financial news data...")
        
        if filename:
            file_path = self.raw_data_dir / filename
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                print(f"File {filename} not found in {self.raw_data_dir}")
                return None
        
        # Look for common news data file patterns
        news_patterns = [
            "raw_analyst_ratings.csv",
            "financial_news*.csv",
            "news_data*.csv",
            "*news*.csv"
        ]
        
        for pattern in news_patterns:
            files = list(self.raw_data_dir.glob(pattern))
            if files:
                print(f"Found news data file: {files[0]}")
                return pd.read_csv(files[0])
        
        print("No news data file found. Please check your data directory.")
        return None
    
    def load_stock_data(self, stock_symbol=None):
        """
        Load historical stock price data
        
        Args:
            stock_symbol (str): Specific stock symbol to load, if None loads all available
            
        Returns:
            dict or pd.DataFrame: Stock data
        """
        print("Loading stock price data...")
        
        if not self.yfinance_data_dir.exists():
            print(f"Stock data directory {self.yfinance_data_dir} not found.")
            return None
        
        stock_files = list(self.yfinance_data_dir.glob("*_historical_data.csv"))
        
        if not stock_files:
            print("No stock data files found.")
            return None
        
        if stock_symbol:
            # Load specific stock
            target_file = self.yfinance_data_dir / f"{stock_symbol}_historical_data.csv"
            if target_file.exists():
                df = pd.read_csv(target_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df['Stock'] = stock_symbol
                return df
            else:
                print(f"Stock data for {stock_symbol} not found.")
                return None
        else:
            # Load all stock data
            all_stock_data = {}
            for file in stock_files:
                stock_symbol = file.stem.replace('_historical_data', '')
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(df['Date'])
                df['Stock'] = stock_symbol
                all_stock_data[stock_symbol] = df
                
            print(f"Loaded data for {len(all_stock_data)} stocks: {list(all_stock_data.keys())}")
            return all_stock_data
    
    def preprocess_news_data(self, df):
        """
        Preprocess news data for analysis
        
        Args:
            df (pd.DataFrame): Raw news data
            
        Returns:
            pd.DataFrame: Preprocessed news data
        """
        print("Preprocessing news data...")
        
        if df is None:
            return None
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Standardize column names (handle different possible column names)
        column_mapping = {
            'Headline': 'headline',
            'Title': 'headline',
            'URL': 'url',
            'Link': 'url',
            'Publisher': 'publisher',
            'Source': 'publisher',
            'Date': 'date',
            'Timestamp': 'date',
            'Stock': 'stock',
            'Symbol': 'stock',
            'Ticker': 'stock'
        }
        
        processed_df = processed_df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['headline', 'date', 'stock']
        missing_columns = [col for col in required_columns if col not in processed_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            return None
        
        # Clean and convert data types
        try:
            # Convert date column
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            
            # Remove rows with invalid dates
            processed_df = processed_df.dropna(subset=['date'])
            
            # Clean headline text
            processed_df['headline'] = processed_df['headline'].astype(str)
            processed_df['headline'] = processed_df['headline'].str.strip()
            
            # Clean stock symbols
            processed_df['stock'] = processed_df['stock'].astype(str).str.upper().str.strip()
            
            # Remove duplicates
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(subset=['headline', 'date', 'stock'])
            final_count = len(processed_df)
            
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicate rows")
            
            # Sort by date
            processed_df = processed_df.sort_values('date').reset_index(drop=True)
            
            print(f"Preprocessing completed. Final dataset shape: {processed_df.shape}")
            print(f"Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
            print(f"Unique stocks: {processed_df['stock'].nunique()}")
            
            return processed_df
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
    
    def merge_news_and_stock_data(self, news_df, stock_data_dict):
        """
        Merge news data with corresponding stock price data
        
        Args:
            news_df (pd.DataFrame): Preprocessed news data
            stock_data_dict (dict): Dictionary of stock DataFrames
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        print("Merging news and stock data...")
        
        if news_df is None or stock_data_dict is None:
            print("Cannot merge: missing news or stock data")
            return None
        
        merged_data = []
        
        for stock_symbol, stock_df in stock_data_dict.items():
            # Get news for this stock
            stock_news = news_df[news_df['stock'] == stock_symbol].copy()
            
            if len(stock_news) == 0:
                continue
            
            # Prepare stock data
            stock_df_clean = stock_df.copy()
            stock_df_clean['Date'] = pd.to_datetime(stock_df_clean['Date'])
            stock_df_clean = stock_df_clean.sort_values('Date')
            
            # Calculate daily returns
            stock_df_clean['Daily_Return'] = stock_df_clean['Close'].pct_change()
            stock_df_clean['Price_Change'] = stock_df_clean['Close'].diff()
            stock_df_clean['Volume_Change'] = stock_df_clean['Volume'].pct_change()
            
            # Calculate technical indicators
            stock_df_clean['SMA_5'] = stock_df_clean['Close'].rolling(window=5).mean()
            stock_df_clean['SMA_20'] = stock_df_clean['Close'].rolling(window=20).mean()
            stock_df_clean['Volatility'] = stock_df_clean['Daily_Return'].rolling(window=20).std()
            
            # Merge with news data
            stock_news['news_date'] = stock_news['date'].dt.date
            stock_df_clean['stock_date'] = stock_df_clean['Date'].dt.date
            
            # Merge on date
            merged_stock = pd.merge(
                stock_news,
                stock_df_clean,
                left_on='news_date',
                right_on='stock_date',
                how='left'
            )
            
            # For news without exact date match, use forward fill for stock data
            merged_stock = merged_stock.sort_values('date')
            stock_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
                           'Price_Change', 'Volume_Change', 'SMA_5', 'SMA_20', 'Volatility']
            
            for col in stock_columns:
                if col in merged_stock.columns:
                    merged_stock[col] = merged_stock[col].fillna(method='ffill')
            
            merged_data.append(merged_stock)
        
        if not merged_data:
            print("No data could be merged")
            return None
        
        # Combine all merged data
        final_merged = pd.concat(merged_data, ignore_index=True)
        
        # Clean up columns
        final_merged = final_merged.drop(['news_date', 'stock_date'], axis=1, errors='ignore')
        
        print(f"Merge completed. Final dataset shape: {final_merged.shape}")
        print(f"Stocks with merged data: {final_merged['stock'].nunique()}")
        
        return final_merged
    
    def save_processed_data(self, df, filename="processed_financial_data.csv"):
        """
        Save processed data to file
        
        Args:
            df (pd.DataFrame): Processed data to save
            filename (str): Output filename
        """
        if df is None:
            print("No data to save")
            return
        
        output_path = self.processed_data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        
        # Save summary statistics
        summary_path = self.processed_data_dir / f"summary_{filename.replace('.csv', '.txt')}"
        with open(summary_path, 'w') as f:
            f.write("PROCESSED DATA SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
            f.write(f"Unique stocks: {df['stock'].nunique()}\n")
            f.write(f"Total articles: {len(df)}\n")
            f.write("\nColumn info:\n")
            f.write(str(df.dtypes))
            f.write("\n\nMissing values:\n")
            f.write(str(df.isnull().sum()))
        
        print(f"Summary saved to: {summary_path}")
    
    def load_and_preprocess_all(self):
        """
        Complete data loading and preprocessing pipeline
        
        Returns:
            pd.DataFrame: Fully processed and merged dataset
        """
        print("Starting complete data loading and preprocessing pipeline...")
        print("=" * 60)
        
        # Step 1: Load news data
        news_df = self.load_news_data()
        if news_df is None:
            print("Failed to load news data. Exiting.")
            return None
        
        # Step 2: Preprocess news data
        processed_news = self.preprocess_news_data(news_df)
        if processed_news is None:
            print("Failed to preprocess news data. Exiting.")
            return None
        
        # Step 3: Load stock data
        stock_data = self.load_stock_data()
        if stock_data is None:
            print("No stock data available. Returning news data only.")
            self.save_processed_data(processed_news, "processed_news_only.csv")
            return processed_news
        
        # Step 4: Merge data
        merged_data = self.merge_news_and_stock_data(processed_news, stock_data)
        if merged_data is None:
            print("Failed to merge data. Returning news data only.")
            self.save_processed_data(processed_news, "processed_news_only.csv")
            return processed_news
        
        # Step 5: Save processed data
        self.save_processed_data(merged_data, "processed_financial_data_merged.csv")
        
        print("=" * 60)
        print("Data loading and preprocessing completed successfully!")
        
        return merged_data

# Usage example and standalone script
if __name__ == "__main__":
    print("Financial Data Loader and Preprocessor")
    print("Nova Financial Solutions - Week 1 Challenge")
    print("=" * 60)
    
    # Initialize the data loader with correct path
    # Use "../data" to go up one level from notebooks to project root
    loader = FinancialDataLoader(data_directory="../data")
    
    
    if processed_data is not None:
        print("\nDataset ready for EDA and sentiment analysis!")
        print("You can now use this processed data with the EDA script.")
        
        # Quick preview
        print("\nQuick data preview:")
        print(processed_data.head())
        print(f"\nDataset info:")
        print(f"Shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
    else:
        print("Data loading failed. Please check your data files and directory structure.")