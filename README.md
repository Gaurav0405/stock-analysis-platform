# 📈 Stock Analysis Platform

A comprehensive full-stack stock market analysis platform that combines machine learning, technical analysis, sentiment analysis, computer vision, and quantitative finance to provide data-driven market insights. The platform forecasts stock prices, analyzes financial news sentiment, extracts price data from stock chart images, and evaluates trading strategies through historical backtesting and Monte Carlo simulations.

---

## ✨ Key Features

### 📊 Multi-Horizon Stock Forecasting
- Predicts stock prices for **1, 5, 10, and 30-day** horizons.
- Ensemble forecasting using:
  - XGBoost Regressor
  - Random Forest Regressor
  - Linear Regression
  - LSTM (TensorFlow/Keras)

### 📈 Technical Analysis
Implements 11+ technical indicators including:
- Relative Strength Index (RSI)
- MACD
- Bollinger Bands
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Average True Range (ATR)
- Stochastic Oscillator
- On-Balance Volume (OBV)
- Support & Resistance Detection
- Trend Analysis
- Volatility Indicators

### 📰 Financial News Sentiment Analysis
- Collects financial headlines from Yahoo Finance and Google News RSS feeds.
- Performs sentiment analysis using VADER.
- Classifies market sentiment as:
  - 🟢 Bullish
  - 🔴 Bearish
  - ⚪ Neutral

### 🖼️ Chart Image Analysis
Upload a stock chart image and the platform automatically:
- Detects chart lines using OpenCV
- Extracts price labels using Tesseract OCR
- Maps pixel coordinates to price values
- Reconstructs historical price series
- Generates synthetic OHLCV data
- Executes the forecasting pipeline

### 📉 Risk & Performance Analysis
- Monte Carlo Simulation
- Historical Backtesting
- Strategy Evaluation
- Sharpe Ratio
- Maximum Drawdown
- Win Rate

---

# 📊 Project Highlights

| Feature | Description |
|---------|-------------|
| Historical Dataset | 5 years of historical stock market data |
| Forecast Horizons | 1, 5, 10, and 30-day predictions |
| Predictive Models | Ensemble of Linear Regression, Random Forest, XGBoost, LSTM, and KNN/SVR |
| Technical Analysis | 11+ technical indicators |
| Market Context | S&P 500 and VIX integration |
| News Analysis | Financial news sentiment using RSS feeds and VADER |
| Computer Vision | OpenCV + Tesseract OCR for chart digitization |
| Risk Analysis | 1,000 Monte Carlo simulations |
| Strategy Evaluation | Historical backtesting with risk metrics |

---

# 🏗️ System Architecture

```text
                     User
                       │
                       ▼
               React Frontend
                       │
                Flask REST API
                       │
      ┌────────────────┼────────────────┐
      │                │                │
      ▼                ▼                ▼
 Market Data     News Analysis    Chart Upload
(Yahoo Finance)   (RSS Feeds)        (Image)
      │                │                │
      ▼                ▼                ▼
 Feature        Sentiment Score    OpenCV + OCR
Engineering             │                │
      └─────────────────┼────────────────┘
                        ▼
               Prediction Pipeline
                        │
      ┌─────────────────┼──────────────────┐
      ▼                 ▼                  ▼
 Price Forecast   Direction Model   Monte Carlo &
                                   Backtesting
                        │
                        ▼
                Analysis Dashboard
```

---

# 🛠️ Tech Stack

### Backend
- Python
- Flask
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- XGBoost
- TensorFlow / Keras

### Computer Vision
- OpenCV
- Tesseract OCR
- SciPy

### Frontend
- React
- Vite

### Data Sources
- Yahoo Finance
- Yahoo Finance RSS
- Google News RSS

---

# 📂 Project Structure

```text
Stock-Analysis-Platform/
│
├── analysis/
│   ├── image_analysis.py
│   ├── sentiment.py
│   └── technical.py
│
├── client/
├── data/
├── models/
│   ├── regression.py
│   ├── classification.py
│   └── backtest.py
│
├── utils/
├── visualization/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/Gaurav0405/stock-analysis-platform.git
cd stock-analysis-platform
```

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Backend

```bash
python app.py
```

## Run the Frontend

```bash
cd client
npm install
npm run dev
```

---

# 🤖 Machine Learning Pipeline

### Regression Models
- XGBoost Regressor
- Random Forest Regressor
- Linear Regression
- LSTM

### Classification
- Random Forest Classifier

### Ensemble Strategy
The platform combines predictions from multiple regression models to improve robustness and reduce overfitting. Model weights are dynamically adjusted based on market conditions, and a volatility cap is applied to prevent unrealistic price forecasts.

---

# 🖼️ Chart Image Processing Pipeline

1. Upload a stock chart image.
2. Detect chart lines using HSV color segmentation.
3. Smooth extracted points using spline interpolation.
4. Extract price labels using Tesseract OCR.
5. Convert pixel coordinates into actual price values.
6. Generate synthetic OHLCV data.
7. Execute the forecasting pipeline.

---

# 📈 Evaluation Framework

The platform includes an evaluation pipeline for measuring forecasting performance and trading strategy effectiveness using industry-standard metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Directional Accuracy
- Sharpe Ratio
- Maximum Drawdown
- Win Rate

These metrics can be generated for any supported stock using the project's evaluation and backtesting modules.

---

# 🚀 Future Enhancements

- Real-time market streaming
- Transformer-based forecasting models
- Financial domain language models
- Portfolio optimization
- Multi-stock comparison
- Docker support
- Cloud deployment
- Automated model retraining

---

# 🤝 Contributing

Contributions, feature requests, and bug reports are welcome. Feel free to fork the repository and submit a pull request.

---

# 👨‍💻 Author

**Gaurav Jain**

- GitHub: https://github.com/Gaurav0405
- LinkedIn: https://www.linkedin.com/in/gaurav-jain-b5b249278/
