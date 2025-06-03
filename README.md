# Nova Financial Analysis - Predicting Price Moves with News Sentiment

## Project Overview
This project analyzes correlations between financial news sentiment and stock market movements using advanced data analysis techniques.

## Business Objective
Enhance predictive analytics capabilities for financial forecasting through:
- Sentiment analysis of financial news headlines
- Correlation analysis between news sentiment and stock price movements
- Development of predictive investment strategies

## Dataset
- **FNSPID**: Financial News and Stock Price Integration Dataset
- **Stock Data**: Historical price data for AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA
- **News Data**: Headlines with timestamps, publishers, and stock symbols

## Setup Instructions
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run notebooks in order: EDA → Technical Indicators → Sentiment Correlation

## Project Structure
- `src/`: Core Python modules
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Raw and processed datasets
- `tests/`: Unit tests
- `scripts/`: Utility scripts

## Key Technologies
- yfinance: Stock data retrieval
- TA-Lib: Technical indicators
- TextBlob/NLTK: Sentiment analysis
- Pandas/NumPy: Data manipulation
- Matplotlib/Plotly: Visualization

## Tasks Completed
- [x] Task 1: Git and GitHub setup
- [ ] Task 2: Quantitative analysis with PyNance and TA-Lib
- [ ] Task 3: Correlation between news and stock movement
