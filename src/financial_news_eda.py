# Financial News Sentiment Analysis - Exploratory Data Analysis
# Nova Financial Solutions - Week 1 Challenge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
from wordcloud import WordCloud

# Topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Time series analysis
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialNewsEDA:
    """
    Comprehensive EDA class for Financial News Data Analysis
    """
    
    def __init__(self, data_path):
        """
        Initialize the EDA class with data loading
        """
        self.data_path = data_path
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """
        Load the financial news dataset
        """
        try:
            # Assuming the main news data is in a CSV file
            # Adjust the path based on your actual data file
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def basic_info(self):
        """
        Display basic information about the dataset
        """
        print("="*50)
        print("DATASET BASIC INFORMATION")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
        print("\nColumn data types:")
        print(self.df.dtypes)
        
        print("\nMissing values:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\nFirst few rows:")
        print(self.df.head())
        
    def descriptive_statistics(self):
        """
        Perform descriptive statistics analysis
        """
        print("="*50)
        print("DESCRIPTIVE STATISTICS")
        print("="*50)
        
        # Headline length analysis
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['headline_word_count'] = self.df['headline'].str.split().str.len()
        
        print("Headline Length Statistics:")
        print(self.df['headline_length'].describe())
        
        print("\nHeadline Word Count Statistics:")
        print(self.df['headline_word_count'].describe())
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Headline length distribution
        axes[0,0].hist(self.df['headline_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Headline Lengths')
        axes[0,0].set_xlabel('Character Count')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df['headline_length'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,0].legend()
        
        # Word count distribution
        axes[0,1].hist(self.df['headline_word_count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Distribution of Headline Word Counts')
        axes[0,1].set_xlabel('Word Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(self.df['headline_word_count'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,1].legend()
        
        # Box plot for headline lengths
        axes[1,0].boxplot(self.df['headline_length'], vert=True)
        axes[1,0].set_title('Headline Length Box Plot')
        axes[1,0].set_ylabel('Character Count')
        
        # Articles per publisher (top 20)
        publisher_counts = self.df['publisher'].value_counts().head(20)
        axes[1,1].barh(range(len(publisher_counts)), publisher_counts.values)
        axes[1,1].set_yticks(range(len(publisher_counts)))
        axes[1,1].set_yticklabels(publisher_counts.index, fontsize=8)
        axes[1,1].set_title('Top 20 Publishers by Article Count')
        axes[1,1].set_xlabel('Number of Articles')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'headline_length_stats': self.df['headline_length'].describe(),
            'word_count_stats': self.df['headline_word_count'].describe(),
            'publisher_counts': self.df['publisher'].value_counts()
        }
    
    def publisher_analysis(self):
        """
        Analyze publisher patterns and characteristics
        """
        print("="*50)
        print("PUBLISHER ANALYSIS")
        print("="*50)
        
        # Publisher statistics
        total_publishers = self.df['publisher'].nunique()
        print(f"Total unique publishers: {total_publishers}")
        
        publisher_stats = self.df['publisher'].value_counts()
        print(f"\nTop 10 most active publishers:")
        print(publisher_stats.head(10))
        
        # Identify email-based publishers
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.df['is_email_publisher'] = self.df['publisher'].str.contains(email_pattern, na=False)
        email_publishers = self.df[self.df['is_email_publisher']]
        
        print(f"\nEmail-based publishers: {email_publishers['publisher'].nunique()}")
        if len(email_publishers) > 0:
            # Extract domains from email publishers
            email_publishers['domain'] = email_publishers['publisher'].str.extract(r'@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})')
            domain_counts = email_publishers['domain'].value_counts()
            print("Top domains from email publishers:")
            print(domain_counts.head(10))
        
        # Publisher diversity metrics
        publisher_article_counts = self.df['publisher'].value_counts()
        
        # Gini coefficient for publisher concentration
        def gini_coefficient(x):
            """Calculate Gini coefficient"""
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        gini = gini_coefficient(publisher_article_counts.values)
        print(f"\nPublisher concentration (Gini coefficient): {gini:.3f}")
        print("(0 = perfectly equal distribution, 1 = maximum inequality)")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Publisher concentration
        top_publishers = publisher_stats.head(15)
        axes[0,0].bar(range(len(top_publishers)), top_publishers.values)
        axes[0,0].set_xticks(range(len(top_publishers)))
        axes[0,0].set_xticklabels(top_publishers.index, rotation=45, ha='right', fontsize=8)
        axes[0,0].set_title('Top 15 Publishers by Article Count')
        axes[0,0].set_ylabel('Number of Articles')
        
        # Publisher distribution (log scale)
        axes[0,1].hist(publisher_stats.values, bins=50, alpha=0.7, color='orange')
        axes[0,1].set_yscale('log')
        axes[0,1].set_title('Distribution of Articles per Publisher (Log Scale)')
        axes[0,1].set_xlabel('Number of Articles')
        axes[0,1].set_ylabel('Number of Publishers (Log)')
        
        # Cumulative percentage of articles by publishers
        cumulative_articles = np.cumsum(publisher_stats.values)
        cumulative_percentage = (cumulative_articles / cumulative_articles[-1]) * 100
        axes[1,0].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
        axes[1,0].set_title('Cumulative Percentage of Articles by Publishers')
        axes[1,0].set_xlabel('Publisher Rank')
        axes[1,0].set_ylabel('Cumulative Percentage of Articles')
        axes[1,0].grid(True, alpha=0.3)
        
        # Email vs non-email publishers
        email_vs_nonemail = self.df['is_email_publisher'].value_counts()
        axes[1,1].pie(email_vs_nonemail.values, labels=['Non-Email', 'Email'], autopct='%1.1f%%')
        axes[1,1].set_title('Email vs Non-Email Publishers')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'total_publishers': total_publishers,
            'publisher_stats': publisher_stats,
            'gini_coefficient': gini,
            'email_publishers_count': email_publishers['publisher'].nunique() if len(email_publishers) > 0 else 0
        }
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def text_analysis_topic_modeling(self):
        """
        Perform text analysis and topic modeling
        """
        print("="*50)
        print("TEXT ANALYSIS & TOPIC MODELING")
        print("="*50)
        
        # Preprocess headlines
        print("Preprocessing headlines...")
        self.df['processed_headline'] = self.df['headline'].apply(self.preprocess_text)
        
        # Most common words
        all_words = ' '.join(self.df['processed_headline']).split()
        word_freq = Counter(all_words)
        most_common_words = word_freq.most_common(30)
        
        print("Top 30 most common words:")
        for word, freq in most_common_words:
            print(f"{word}: {freq}")
        
        # Financial keywords analysis
        financial_keywords = [
            'stock', 'price', 'target', 'buy', 'sell', 'earnings', 'revenue',
            'profit', 'loss', 'dividend', 'acquisition', 'merger', 'ipo',
            'fda', 'approval', 'trial', 'upgrade', 'downgrade', 'analyst',
            'forecast', 'outlook', 'guidance', 'beat', 'miss', 'estimate'
        ]
        
        keyword_counts = {}
        for keyword in financial_keywords:
            keyword_counts[keyword] = self.df['processed_headline'].str.contains(keyword).sum()
        
        print(f"\nFinancial keywords frequency:")
        sorted_keywords = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
        for keyword, count in sorted_keywords.items():
            if count > 0:
                print(f"{keyword}: {count}")
        
        # Topic Modeling with LDA
        print("\nPerforming topic modeling...")
        
        # Vectorize the text
        vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
        doc_term_matrix = vectorizer.fit_transform(self.df['processed_headline'])
        
        # LDA Topic Modeling
        n_topics = 8
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        print(f"\nIdentified {n_topics} topics:")
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            print(f"Topic {topic_idx + 1}: {', '.join(top_words[:5])}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Word frequency bar chart
        words, frequencies = zip(*most_common_words[:20])
        axes[0,0].barh(range(len(words)), frequencies)
        axes[0,0].set_yticks(range(len(words)))
        axes[0,0].set_yticklabels(words)
        axes[0,0].set_title('Top 20 Most Common Words')
        axes[0,0].set_xlabel('Frequency')
        
        # Financial keywords
        fin_words = [k for k, v in sorted_keywords.items() if v > 0][:15]
        fin_counts = [sorted_keywords[k] for k in fin_words]
        axes[0,1].bar(range(len(fin_words)), fin_counts)
        axes[0,1].set_xticks(range(len(fin_words)))
        axes[0,1].set_xticklabels(fin_words, rotation=45, ha='right')
        axes[0,1].set_title('Financial Keywords Frequency')
        axes[0,1].set_ylabel('Count')
        
        # Word cloud
        wordcloud_text = ' '.join(self.df['processed_headline'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(wordcloud_text)
        axes[1,0].imshow(wordcloud, interpolation='bilinear')
        axes[1,0].axis('off')
        axes[1,0].set_title('Word Cloud of Headlines')
        
        # Topic distribution
        doc_topic_probs = lda.transform(doc_term_matrix)
        topic_prevalence = doc_topic_probs.mean(axis=0)
        axes[1,1].bar(range(1, n_topics + 1), topic_prevalence)
        axes[1,1].set_title('Topic Prevalence Distribution')
        axes[1,1].set_xlabel('Topic Number')
        axes[1,1].set_ylabel('Average Probability')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'most_common_words': most_common_words,
            'financial_keywords': sorted_keywords,
            'topics': topics,
            'topic_prevalence': topic_prevalence
        }
    
    def time_series_analysis(self):
        """
        Perform time series analysis on publication patterns
        """
        print("="*50)
        print("TIME SERIES ANALYSIS")
        print("="*50)
        
        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Extract time components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['is_weekend'] = self.df['date'].dt.weekday >= 5
        
        # Basic time statistics
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Total time span: {(self.df['date'].max() - self.df['date'].min()).days} days")
        
        # Daily publication frequency
        daily_counts = self.df.groupby(self.df['date'].dt.date).size()
        print(f"\nDaily publication statistics:")
        print(daily_counts.describe())
        
        # Weekly patterns
        weekly_pattern = self.df['day_of_week'].value_counts()
        print(f"\nWeekly publication pattern:")
        print(weekly_pattern)
        
        # Hourly patterns
        hourly_pattern = self.df['hour'].value_counts().sort_index()
        print(f"\nHourly publication pattern:")
        print(hourly_pattern.head(10))
        
        # Weekend vs weekday
        weekend_vs_weekday = self.df['is_weekend'].value_counts()
        print(f"\nWeekend vs Weekday publications:")
        print(f"Weekday: {weekend_vs_weekday[False]}")
        print(f"Weekend: {weekend_vs_weekday[True]}")
        
        # Visualizations
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Daily time series
        daily_counts.plot(ax=axes[0,0], title='Daily Publication Frequency', color='blue', alpha=0.7)
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Articles')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Monthly aggregation
        monthly_counts = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).size()
        monthly_counts.plot(ax=axes[0,1], title='Monthly Publication Frequency', color='green', alpha=0.7)
        axes[0,1].set_xlabel('Year-Month')
        axes[0,1].set_ylabel('Number of Articles')
        
        # Day of week pattern
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_ordered = weekly_pattern.reindex(day_order)
        axes[1,0].bar(range(7), weekly_ordered.values, color='orange', alpha=0.7)
        axes[1,0].set_xticks(range(7))
        axes[1,0].set_xticklabels(day_order, rotation=45)
        axes[1,0].set_title('Publications by Day of Week')
        axes[1,0].set_ylabel('Number of Articles')
        
        # Hourly pattern
        axes[1,1].bar(hourly_pattern.index, hourly_pattern.values, color='red', alpha=0.7)
        axes[1,1].set_title('Publications by Hour of Day')
        axes[1,1].set_xlabel('Hour')
        axes[1,1].set_ylabel('Number of Articles')
        
        # Rolling average (7-day)
        rolling_avg = daily_counts.rolling(window=7, center=True).mean()
        axes[2,0].plot(daily_counts.index, daily_counts.values, alpha=0.3, color='blue', label='Daily')
        axes[2,0].plot(rolling_avg.index, rolling_avg.values, color='red', linewidth=2, label='7-day MA')
        axes[2,0].set_title('Daily Publications with 7-day Moving Average')
        axes[2,0].set_xlabel('Date')
        axes[2,0].set_ylabel('Number of Articles')
        axes[2,0].legend()
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # Heatmap of publications by day and hour
        hour_day_pivot = self.df.pivot_table(values='headline', 
                                            index='day_of_week', 
                                            columns='hour', 
                                            aggfunc='count', 
                                            fill_value=0)
        hour_day_pivot = hour_day_pivot.reindex(day_order)
        
        sns.heatmap(hour_day_pivot, ax=axes[2,1], cmap='YlOrRd', cbar_kws={'label': 'Number of Articles'})
        axes[2,1].set_title('Publication Heatmap: Day vs Hour')
        axes[2,1].set_xlabel('Hour of Day')
        axes[2,1].set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.show()
        
        # Detect spikes and anomalies
        spike_threshold = daily_counts.mean() + 2 * daily_counts.std()
        spikes = daily_counts[daily_counts > spike_threshold]
        
        print(f"\nDetected publication spikes (> {spike_threshold:.1f} articles/day):")
        for date, count in spikes.items():
            print(f"{date}: {count} articles")
        
        return {
            'daily_stats': daily_counts.describe(),
            'weekly_pattern': weekly_pattern,
            'hourly_pattern': hourly_pattern,
            'spikes': spikes,
            'weekend_weekday_ratio': weekend_vs_weekday[True] / weekend_vs_weekday[False]
        }
    
    def stock_symbol_analysis(self):
        """
        Analyze stock symbol patterns and coverage
        """
        print("="*50)
        print("STOCK SYMBOL ANALYSIS")
        print("="*50)
        
        # Stock symbol statistics
        stock_counts = self.df['stock'].value_counts()
        total_stocks = self.df['stock'].nunique()
        
        print(f"Total unique stock symbols: {total_stocks}")
        print(f"Top 15 most covered stocks:")
        print(stock_counts.head(15))
        
        # Coverage distribution
        coverage_stats = stock_counts.describe()
        print(f"\nStock coverage statistics:")
        print(coverage_stats)
        
        # Time series by top stocks
        top_stocks = stock_counts.head(10).index
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top stocks coverage
        axes[0,0].barh(range(len(stock_counts.head(15))), stock_counts.head(15).values)
        axes[0,0].set_yticks(range(len(stock_counts.head(15))))
        axes[0,0].set_yticklabels(stock_counts.head(15).index)
        axes[0,0].set_title('Top 15 Most Covered Stocks')
        axes[0,0].set_xlabel('Number of Articles')
        
        # Stock coverage distribution
        axes[0,1].hist(stock_counts.values, bins=50, alpha=0.7, color='green')
        axes[0,1].set_title('Distribution of Articles per Stock')
        axes[0,1].set_xlabel('Number of Articles')
        axes[0,1].set_ylabel('Number of Stocks')
        axes[0,1].set_yscale('log')
        
        # Daily coverage for top 5 stocks
        for i, stock in enumerate(top_stocks[:5]):
            stock_daily = self.df[self.df['stock'] == stock].groupby(self.df['date'].dt.date).size()
            axes[1,0].plot(stock_daily.index, stock_daily.values, alpha=0.7, label=stock)
        
        axes[1,0].set_title('Daily Coverage - Top 5 Stocks')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Number of Articles')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Stock coverage concentration
        cumulative_coverage = np.cumsum(stock_counts.values)
        cumulative_percentage = (cumulative_coverage / cumulative_coverage[-1]) * 100
        axes[1,1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
        axes[1,1].set_title('Cumulative Coverage Percentage by Stock Rank')
        axes[1,1].set_xlabel('Stock Rank')
        axes[1,1].set_ylabel('Cumulative Percentage of Articles')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'total_stocks': total_stocks,
            'stock_counts': stock_counts,
            'coverage_stats': coverage_stats
        }
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary of all analyses
        """
        print("="*60)
        print("COMPREHENSIVE EDA SUMMARY REPORT")
        print("="*60)
        
        # Perform all analyses
        desc_stats = self.descriptive_statistics()
        pub_analysis = self.publisher_analysis()
        text_analysis = self.text_analysis_topic_modeling()
        time_analysis = self.time_series_analysis()
        stock_analysis = self.stock_symbol_analysis()
        
        print("\n" + "="*60)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        print("\n1. DATA QUALITY & SCALE:")
        print(f"   • Dataset contains {len(self.df):,} articles")
        print(f"   • Covers {stock_analysis['total_stocks']} unique stocks")
        print(f"   • Published by {pub_analysis['total_publishers']} different sources")
        print(f"   • Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        
        print("\n2. CONTENT CHARACTERISTICS:")
        print(f"   • Average headline length: {desc_stats['headline_length_stats']['mean']:.1f} characters")
        print(f"   • Average word count: {desc_stats['word_count_stats']['mean']:.1f} words")
        print(f"   • Most active publisher: {pub_analysis['publisher_stats'].index[0]} ({pub_analysis['publisher_stats'].iloc[0]} articles)")
        
        print("\n3. TEMPORAL PATTERNS:")
        print(f"   • Peak publication day: {time_analysis['weekly_pattern'].index[0]}")
        print(f"   • Weekend vs Weekday ratio: {time_analysis['weekend_weekday_ratio']:.2f}")
        print(f"   • Publication spikes detected: {len(time_analysis['spikes'])} days")
        
        print("\n4. CONTENT THEMES:")
        print("   • Top financial keywords found in headlines")
        print("   • Topic modeling reveals distinct themes in financial news")
        print("   • Strong focus on earnings, price targets, and analyst recommendations")
        
        print("\n5. RECOMMENDATIONS FOR SENTIMENT ANALYSIS:")
        print("   • Focus on headlines with financial keywords for better sentiment correlation")
        print("   • Consider temporal patterns when analyzing market impact")
        print("   • Weight analysis by publisher credibility and coverage frequency")
        print("   • Account for publication timing in trading hour analysis")
        
        return {
            'descriptive_stats': desc_stats,
            'publisher_analysis': pub_analysis,
            'text_analysis': text_analysis,
            'time_analysis': time_analysis,
            'stock_analysis': stock_analysis
        }

# Usage Example:
if __name__ == "__main__":
    # Example usage - adjust the path to your actual data file
    
    print("Financial News EDA Analysis")
    print("Nova Financial Solutions - Week 1 Challenge")
    print("="*60)
    
    # Initialize the EDA class
    eda = FinancialNewsEDA('../data/raw/raw_analyst_ratings.csv')

    
    # Load the data
    df = eda.load_data()
    
    if df is not None:
        # Display basic information
        eda.basic_info()
        
        # Run comprehensive analysis
        summary = eda.generate_summary_report()
        
        print("\nAnalysis completed successfully!")
        print("All visualizations and statistics have been generated.")
        print("\nNext steps:")
        print("1. Use these insights to guide your sentiment analysis")
        print("2. Focus on high-impact stocks and publishers")
        print("3. Consider temporal patterns in your correlation analysis")
    else:
        print("Please ensure your data file path is correct and the file exists.")
        print("Expected columns: ['headline', 'url', 'publisher', 'date', 'stock']")