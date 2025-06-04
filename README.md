# Financial News Sentiment Analysis

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
5. Generate sample data: `python scripts/create_sample_data.py`
6. Run notebooks in order: EDA → Technical Indicators → Sentiment Correlation

## Project Structure
- `src/`: Core Python modules
- `notebooks/`: Jupyter notebooks for analysis
  - `EDA.ipynb`: Exploratory Data Analysis
  - `financial_analysis.ipynb`: Financial metrics and indicators
  - `correlation_analysis.ipynb`: News sentiment vs stock correlation
- `data/`: Raw and processed datasets
- `tests/`: Unit tests
- `scripts/`: Utility scripts
  - `create_sample_data.py`: Generate sample financial data
  - `correlation_analysis.py`: Correlation analysis functions
  - `sentiment_analysis.py`: News sentiment processing
  - `financial_analysis.py`: Stock analysis utilities

## Key Technologies
- yfinance: Stock data retrieval
- TA-Lib: Technical indicators
- TextBlob/NLTK: Sentiment analysis
- Pandas/NumPy: Data manipulation
- Matplotlib/Plotly: Visualization
- Scipy: Statistical analysis

## Tasks Completed
- [x] Task 1: Git and GitHub setup
- [x] Task 2: Quantitative analysis with PyNance and TA-Lib
- [x] Task 3: Correlation between news and stock movement
  - ✅ Date alignment between news and stock data
  - ✅ Sentiment analysis using TextBlob
  - ✅ Daily stock return calculations
  - ✅ Pearson correlation analysis
  - ✅ Statistical significance testing
  - ✅ Comprehensive visualizations

## Key Findings (Task 3)
- Implemented comprehensive correlation analysis between news sentiment and stock returns
- Created robust date alignment methodology for news and stock data
- Developed sentiment scoring system using TextBlob
- Generated statistical correlation coefficients with p-values
- Built interactive visualizations for correlation insights

## Usage

### Running the Analysis
```bash
# Generate sample data
python scripts/create_sample_data.py

# Run correlation analysis
python scripts/correlation_analysis.py

# Or use the Jupyter notebook
jupyter notebook notebooks/correlation_analysis.ipynb