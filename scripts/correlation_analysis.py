import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

class NewsStockCorrelation:
    """Class for analyzing correlation between news sentiment and stock movements"""

    def __init__(self, news_data_path, stock_data_path):
        """Initialize with paths to news and stock data files

        Args:
            news_data_path (str): Path to processed news data CSV
            stock_data_path (str): Path to stock price data CSV
        """
        self.news_data_path = news_data_path
        self.stock_data_path = stock_data_path
        self.news_df = None
        self.stock_df = None
        self.merged_df = None

    def load_data(self):
        """Load news and stock data from CSV files and align dates"""
        try:
            self.news_df = pd.read_csv(self.news_data_path)
            self.stock_df = pd.read_csv(self.stock_data_path)

            # Convert date columns to datetime with UTC timezone
            self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'], utc=True).dt.date
            self.news_df['date'] = pd.to_datetime(self.news_df['date'], utc=True).dt.date

            # Standardize date and stock column name to 'Date' and 'stock_symbol'
            self.news_df = self.news_df.rename(columns={'date': 'Date'})
            self.news_df = self.news_df.rename(columns={'stock': 'stock_symbol'})

            # # Normalize dates to start of day for consistent alignment
            # self.stock_df['Date'] = self.stock_df['Date'].dt.normalize()
            # self.news_df['Date'] = self.news_df['Date'].dt.normalize()

            # Sort both dataframes by date
            self.stock_df = self.stock_df.sort_values('Date')
            self.news_df = self.news_df.sort_values('Date')

            # Ensure dates overlap between datasets
            min_date = max(self.stock_df['Date'].min(), self.news_df['Date'].min())
            max_date = min(self.stock_df['Date'].max(), self.news_df['Date'].max())

            # Filter news data to match stock trading days
            self.news_df = self.news_df[self.news_df['Date'].isin(self.stock_df['Date'])]

            # Filter stock data to match news dates
            self.stock_df = self.stock_df[self.stock_df['Date'].isin(self.news_df['Date'])]

            # Calculate daily stock returns
            self.stock_df['daily_return'] = self.stock_df.groupby('stock_symbol')['Adj Close'].pct_change()


            print("Data loaded and aligned successfully")
            return self.news_df, self.stock_df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False


    def align_data(self):
        """Align news and stock data by date and stock symbol"""
        if self.news_df is None or self.stock_df is None:
            raise ValueError("News or stock data not loaded. Call load_data() first.")

        # Merge on date and stock symbol
        self.merged_df = pd.merge(self.news_df, self.stock_df,
                                left_on=['Date', 'stock_symbol'], right_on=['Date', 'stock_symbol'],
                                how='inner')
        return self.merged_df

    def calculate_sentiment_scores(self):
        """Calculate sentiment scores for news headlines using TextBlob"""
        if self.news_df is None:
            raise ValueError("News data not loaded. Call load_data() first.")

        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity

        self.news_df['sentiment_score'] = self.news_df['headline'].apply(get_sentiment)

        # Aggregate sentiment scores by date
        daily_sentiment = self.news_df.groupby('Date')['sentiment_score'].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['Date', 'avg_sentiment', 'article_count']

        return daily_sentiment

    def calculate_stock_returns(self):
        """Calculate daily stock returns"""
        if self.stock_df is None:
            raise ValueError("Stock data not loaded. Call load_data() first.")

        # Calculate daily returns
        self.stock_df['daily_return'] = self.stock_df.groupby('stock_symbol')['Close'].pct_change()

        # Aggregate returns by date if multiple stocks
        daily_returns = self.stock_df.groupby('Date')['daily_return'].mean().reset_index()

        return daily_returns





    def calculate_correlation(self):
        # Calculate Pearson correlation coefficient
        correlation_results = self.merged_df.groupby('stock_symbol').apply(
            lambda merged_df: pearsonr(merged_df['sentiment'], merged_df['daily_return'])[0]
        )
        return correlation_results

    def plot_correlation(self, correlation_results):
        # Plot the correlation results
        plt.figure(figsize=(12, 6))
        sns.barplot(x=correlation_results.index, y=correlation_results.values, palette='coolwarm')
        plt.title('Correlation between Sentiment and Stock Returns')
        plt.xlabel('Stock Symbol')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.axhline(0, color='gray', linestyle='--')
        plt.show()

    def scatter_plot(self):
        # Scatter plot of sentiment scores vs. daily stock returns
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='sentiment', y='daily_return', hue='stock_symbol', data=self.merged_df, palette='deep')
        plt.title('Sentiment Scores vs. Daily Stock Returns')
        plt.xlabel('Sentiment Scores')
        plt.ylabel('Daily Stock Returns')
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.show()


