"""
Stock Forecasting — Enhanced Model with Sentiment Analysis & Multi-Day Predictions
==================================================================================
Features:
- News sentiment analysis from Yahoo Finance RSS
- Multi-day predictions (1, 5, 10, 30 days ahead)
- Ensemble blending with confidence intervals
- Technical indicators (RSI, MACD, Bollinger Bands)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import matplotlib.pyplot as plt
import feedparser
import requests
from bs4 import BeautifulSoup
import warnings
import math
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Sentiment analysis will be limited.")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. LSTM model will be skipped.")

warnings.filterwarnings('ignore')


# =============================================================================
# SENTIMENT ANALYSIS MODULE
# =============================================================================

class SentimentAnalyzer:
    """Fetch and analyze news sentiment for a stock ticker."""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
    
    def fetch_yahoo_rss(self, ticker: str, max_items: int = 20) -> list:
        """Fetch news headlines from Yahoo Finance RSS."""
        headlines = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'source': 'Yahoo Finance'
                })
        except Exception as e:
            print(f"Yahoo RSS fetch error: {e}")
        return headlines
    
    def fetch_google_news(self, ticker: str, company_name: str = None, max_items: int = 10) -> list:
        """Fetch news from Google News RSS."""
        headlines = []
        try:
            query = f"{ticker} stock" if not company_name else f"{company_name} stock"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'source': 'Google News'
                })
        except Exception as e:
            print(f"Google News fetch error: {e}")
        return headlines
    
    def score_headline(self, headline: str) -> float:
        """Score a headline using VADER sentiment analyzer."""
        if not self.analyzer:
            return 0.0
        scores = self.analyzer.polarity_scores(headline)
        return scores['compound']  # -1 (negative) to +1 (positive)
    
    def analyze_ticker(self, ticker: str) -> dict:
        """Get comprehensive sentiment analysis for a ticker."""
        # Fetch headlines
        yahoo_headlines = self.fetch_yahoo_rss(ticker)
        google_headlines = self.fetch_google_news(ticker)
        all_headlines = yahoo_headlines + google_headlines
        
        if not all_headlines:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'headline_count': 0,
                'headlines': [],
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 100
            }
        
        # Score all headlines
        scores = []
        scored_headlines = []
        for item in all_headlines:
            score = self.score_headline(item['title'])
            scores.append(score)
            scored_headlines.append({
                **item,
                'sentiment_score': score,
                'sentiment_label': 'POSITIVE' if score > 0.05 else ('NEGATIVE' if score < -0.05 else 'NEUTRAL')
            })
        
        # Calculate aggregates
        avg_score = np.mean(scores)
        positive_count = sum(1 for s in scores if s > 0.05)
        negative_count = sum(1 for s in scores if s < -0.05)
        neutral_count = len(scores) - positive_count - negative_count
        
        # Determine overall label
        if avg_score > 0.1:
            label = 'BULLISH'
        elif avg_score > 0.05:
            label = 'SLIGHTLY BULLISH'
        elif avg_score < -0.1:
            label = 'BEARISH'
        elif avg_score < -0.05:
            label = 'SLIGHTLY BEARISH'
        else:
            label = 'NEUTRAL'
        
        return {
            'sentiment_score': avg_score,
            'sentiment_label': label,
            'headline_count': len(all_headlines),
            'headlines': scored_headlines[:10],  # Top 10 for display
            'positive_pct': (positive_count / len(scores)) * 100,
            'negative_pct': (negative_count / len(scores)) * 100,
            'neutral_pct': (neutral_count / len(scores)) * 100
        }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download and prepare stock data."""
    print(f"\nDownloading {ticker} data...")
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    raw = raw.reset_index()
    
    df = pd.DataFrame({
        "Date": raw["Date"].values,
        "Open": np.array(raw["Open"], dtype=float).reshape(-1),
        "High": np.array(raw["High"], dtype=float).reshape(-1),
        "Low": np.array(raw["Low"], dtype=float).reshape(-1),
        "Close": np.array(raw["Close"], dtype=float).reshape(-1),
        "Volume": np.array(raw["Volume"], dtype=float).reshape(-1),
    })
    
    df.set_index("Date", inplace=True)
    df = df.dropna()
    print(f"Downloaded {len(df)} days of data")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df = df.copy()
    
    # RSI
    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    
    # MACD
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df["Close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["bb_pct"] = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])  # Bollinger %B
    
    # ATR (Average True Range) - volatility
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    
    # Stochastic Oscillator - momentum
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # OBV (On-Balance Volume) - volume trend
    df["obv"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["obv_pct"] = df["obv"].pct_change()  # OBV rate of change
    
    return df


def build_features(df: pd.DataFrame, sentiment_score: float = 0.0, 
                   market_data: dict = None) -> tuple:
    """Build feature set for ML models."""
    d = df.copy()
    
    # Returns
    d["ret_1d"] = d["Close"].pct_change()
    
    # Lagged features
    for lag in [2, 3, 4, 5]:
        d[f"lag_{lag}"] = d["Close"].shift(lag)
    
    # Moving averages and volatility
    for w in [5, 10, 20]:
        d[f"sma_{w}"] = d["Close"].rolling(w).mean()
        d[f"ema_{w}"] = d["Close"].ewm(span=w).mean()
        d[f"vol_{w}"] = d["ret_1d"].rolling(w).std()
    
    # Sentiment
    d["sentiment"] = sentiment_score
    
    # Market context features
    if market_data:
        if "sp500" in market_data:
            d["sp500_ret"] = market_data["sp500"]["ret"].reindex(d.index).fillna(0)
            d["stock_vs_market"] = d["ret_1d"] - d["sp500_ret"]
        if "vix" in market_data:
            d["vix_level"] = market_data["vix"]["close"].reindex(d.index).fillna(method='ffill')
    
    d = d.dropna()
    exclude_cols = ["Target_1", "Target_5", "Target_10", "Target_30", 
                    "Target_High", "Target_Low", "Direction"]
    features = [c for c in d.columns if c not in exclude_cols]
    
    return d, features


def download_market_context() -> dict:
    """Download S&P 500 and VIX data for market context."""
    market_data = {}
    
    try:
        # S&P 500
        sp500 = yf.download("^GSPC", period="5y", interval="1d", progress=False)
        if not sp500.empty:
            sp500_close = np.array(sp500["Close"], dtype=float).reshape(-1)
            market_data["sp500"] = {
                "close": pd.Series(sp500_close, index=sp500.index),
                "ret": pd.Series(sp500_close, index=sp500.index).pct_change()
            }
            print("  ✓ S&P 500 data loaded")
    except Exception as e:
        print(f"  ✗ S&P 500 download failed: {e}")
    
    try:
        # VIX Fear Index
        vix = yf.download("^VIX", period="5y", interval="1d", progress=False)
        if not vix.empty:
            vix_close = np.array(vix["Close"], dtype=float).reshape(-1)
            market_data["vix"] = {
                "close": pd.Series(vix_close, index=vix.index)
            }
            print("  ✓ VIX data loaded")
    except Exception as e:
        print(f"  ✗ VIX download failed: {e}")
    
    return market_data


def create_price_range_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create targets for next-day high and low prediction."""
    df = df.copy()
    df["Target_High"] = df["High"].shift(-1) - df["Close"]  # Change to next high
    df["Target_Low"] = df["Low"].shift(-1) - df["Close"]    # Change to next low
    return df


def create_direction_target(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Create classification target: UP (>1%), DOWN (<-1%), FLAT."""
    df = df.copy()
    next_ret = df["Close"].pct_change().shift(-1)
    
    # 0 = FLAT, 1 = UP, -1 = DOWN
    df["Direction"] = 0
    df.loc[next_ret > threshold, "Direction"] = 1
    df.loc[next_ret < -threshold, "Direction"] = -1
    
    return df


# =============================================================================
# CLASSIFICATION MODE
# =============================================================================

class DirectionClassifier:
    """Predict stock direction: UP, DOWN, or FLAT with confidence."""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the classifier."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        self.rf_model.fit(X_train_s, y_train)
        predictions = self.rf_model.predict(X_test_s)
        
        accuracy = accuracy_score(y_test, predictions)
        self.trained = True
        
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "y_test": y_test
        }
    
    def predict_proba(self, X):
        """Get prediction with confidence percentages."""
        if not self.trained:
            return None
        
        X_s = self.scaler.transform(X)
        proba = self.rf_model.predict_proba(X_s)[0]
        classes = self.rf_model.classes_
        
        result = {
            "prediction": self.rf_model.predict(X_s)[0],
            "confidence": {}
        }
        
        for i, cls in enumerate(classes):
            label = "UP" if cls == 1 else ("DOWN" if cls == -1 else "FLAT")
            result["confidence"][label] = proba[i] * 100
        
        return result


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """Backtest a trading strategy based on model predictions."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                     confidence_threshold: float = 0.6, 
                     direction_predictions: np.ndarray = None,
                     direction_confidence: np.ndarray = None) -> dict:
        """
        Run backtest with simple strategy:
        - Buy when predicted change > 0 (or UP with high confidence)
        - Sell/short when predicted change < 0 (or DOWN with high confidence)
        """
        capital = self.initial_capital
        position = 0  # 1 = long, -1 = short, 0 = flat
        entry_price = 0
        trades = []
        equity = [capital]
        
        closes = df["Close"].values
        
        for i in range(len(predictions)):
            if i >= len(closes) - 1:
                break
                
            current_price = closes[i]
            next_price = closes[i + 1]
            pred = predictions[i]
            
            # Determine signal
            if direction_predictions is not None and direction_confidence is not None:
                conf = direction_confidence[i] if i < len(direction_confidence) else 0.5
                dir_pred = direction_predictions[i] if i < len(direction_predictions) else 0
                
                if dir_pred == 1 and conf >= confidence_threshold:
                    signal = 1  # Buy
                elif dir_pred == -1 and conf >= confidence_threshold:
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold
            else:
                # Use regression predictions
                if pred > 0:
                    signal = 1
                elif pred < 0:
                    signal = -1
                else:
                    signal = 0
            
            # Execute trades
            if signal == 1 and position <= 0:
                # Close short if any, go long
                if position == -1:
                    pnl = (entry_price - current_price) * (capital / entry_price)
                    capital += pnl
                    trades.append({"type": "close_short", "price": current_price, "pnl": pnl})
                
                position = 1
                entry_price = current_price
                trades.append({"type": "long", "price": current_price, "pnl": 0})
                
            elif signal == -1 and position >= 0:
                # Close long if any, go short
                if position == 1:
                    pnl = (current_price - entry_price) * (capital / entry_price)
                    capital += pnl
                    trades.append({"type": "close_long", "price": current_price, "pnl": pnl})
                
                position = -1
                entry_price = current_price
                trades.append({"type": "short", "price": current_price, "pnl": 0})
            
            # Update equity
            if position == 1:
                unrealized = (next_price - entry_price) * (capital / entry_price)
                equity.append(capital + unrealized)
            elif position == -1:
                unrealized = (entry_price - next_price) * (capital / entry_price)
                equity.append(capital + unrealized)
            else:
                equity.append(capital)
        
        # Close final position
        if position == 1:
            final_pnl = (closes[-1] - entry_price) * (capital / entry_price)
            capital += final_pnl
        elif position == -1:
            final_pnl = (entry_price - closes[-1]) * (capital / entry_price)
            capital += final_pnl
        
        self.trades = trades
        self.equity_curve = equity
        
        # Calculate metrics
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Win rate
        closed_trades = [t for t in trades if t["pnl"] != 0]
        wins = sum(1 for t in closed_trades if t["pnl"] > 0)
        win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
        
        return {
            "final_capital": capital,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "total_trades": len(trades),
            "equity_curve": equity
        }
    
    def plot_equity_curve(self, ticker: str):
        """Plot equity curve."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.equity_curve,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title=f'{ticker} Backtest Equity Curve',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            height=400
        )
        filepath = f"{OUTPUT_DIR}/{ticker}_backtest.html"
        fig.write_html(filepath)
        print(f"  Saved: {filepath}")





# =============================================================================
# MULTI-DAY PREDICTION MODELS
# =============================================================================

def create_multi_horizon_targets(df: pd.DataFrame, horizons: list = [1, 5, 10, 30]) -> pd.DataFrame:
    """Create target variables for multiple prediction horizons."""
    df = df.copy()
    for h in horizons:
        df[f"Target_{h}"] = df["Close"].shift(-h) - df["Close"]
    return df


def train_models_for_horizon(X_train, y_train, X_test, y_test, horizon: int) -> dict:
    """Train multiple models for a specific prediction horizon."""
    results = {}
    
    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = {
        "model": xgb,
        "predictions": pred_xgb,
        "MAE": mean_absolute_error(y_test, pred_xgb),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred_xgb))
    }
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "model": rf,
        "predictions": pred_rf,
        "MAE": mean_absolute_error(y_test, pred_rf),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred_rf))
    }
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results["LinearRegression"] = {
        "model": lr,
        "predictions": pred_lr,
        "MAE": mean_absolute_error(y_test, pred_lr),
        "RMSE": math.sqrt(mean_squared_error(y_test, pred_lr))
    }
    
    return results


def train_lstm_model(closes: np.ndarray, horizon: int, window: int = 60) -> tuple:
    """Train LSTM model for a specific horizon."""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None
    
    scaler = MinMaxScaler()
    closes_scaled = scaler.fit_transform(closes.reshape(-1, 1))
    
    X, y = [], []
    for i in range(window, len(closes) - horizon):
        X.append(closes_scaled[i-window:i, 0])
        y.append(closes[i + horizon] - closes[i])  # Price change
    
    X = np.array(X)
    y = np.array(y)
    
    cut = int(len(X) * 0.8)
    X_train = X[:cut].reshape(-1, window, 1)
    y_train = y[:cut]
    X_test = X[cut:].reshape(-1, window, 1)
    y_test = y[cut:]
    
    model = Sequential([
        LSTM(64, input_shape=(window, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    
    predictions = model.predict(X_test, verbose=0).flatten()
    
    return model, scaler, {
        "predictions": predictions,
        "y_test": y_test,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": math.sqrt(mean_squared_error(y_test, predictions))
    }


# =============================================================================
# ENSEMBLE & PREDICTION
# =============================================================================

def make_ensemble_prediction(models: dict, latest_features: np.ndarray, 
                              lstm_model=None, lstm_scaler=None, 
                              recent_closes=None, window: int = 60) -> dict:
    """Make ensemble prediction combining all models."""
    predictions = []
    
    # Tabular models
    for name in ["XGBoost", "RandomForest", "LinearRegression"]:
        if name in models:
            pred = models[name]["model"].predict(latest_features)[0]
            predictions.append(pred)
    
    # LSTM
    if lstm_model and lstm_scaler and recent_closes is not None:
        scaled = lstm_scaler.transform(recent_closes[-window:].reshape(-1, 1))
        lstm_input = scaled.reshape(1, window, 1)
        lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0, 0]
        predictions.append(lstm_pred)
    
    if not predictions:
        return {"mean": 0, "std": 0, "predictions": []}
    
    return {
        "mean": np.mean(predictions),
        "std": np.std(predictions),
        "predictions": predictions
    }


def apply_volatility_cap(predicted_change: float, last_close: float, 
                         recent_volatility: float) -> float:
    """Cap predictions based on recent volatility."""
    max_dev_pct = min(0.10, 3 * recent_volatility)  # Max 10% or 3x volatility
    allowed_abs = max_dev_pct * last_close
    
    if predicted_change > allowed_abs:
        return allowed_abs
    elif predicted_change < -allowed_abs:
        return -allowed_abs
    return predicted_change


# =============================================================================
# VISUALIZATION
# =============================================================================

import os
OUTPUT_DIR = "d:/stock/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_price_history(df: pd.DataFrame, ticker: str):
    """Plot price history with technical indicators."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f'{ticker} Price History', 'RSI', 'MACD'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], 
                             line=dict(color='rgba(128,128,128,0.3)'),
                             name="BB High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"],
                             line=dict(color='rgba(128,128,128,0.3)'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                             name="BB Low"), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], 
                             line=dict(color='purple'),
                             name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], 
                             line=dict(color='blue'),
                             name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                             line=dict(color='orange'),
                             name="Signal"), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False,
                      xaxis_rangeslider_visible=False)
    filepath = f"{OUTPUT_DIR}/{ticker}_price_history.html"
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_multi_day_predictions(last_close: float, predictions: dict, 
                               ticker: str, dates: list):
    """Plot multi-day predictions with confidence intervals."""
    horizons = sorted(predictions.keys())
    
    # Build price path
    pred_prices = [last_close]
    upper_prices = [last_close]
    lower_prices = [last_close]
    
    for h in horizons:
        pred = predictions[h]
        pred_price = last_close + pred["mean"]
        pred_prices.append(pred_price)
        upper_prices.append(pred_price + 2 * pred["std"])
        lower_prices.append(pred_price - 2 * pred["std"])
    
    x_labels = ["Today"] + [f"+{h}d" for h in horizons]
    
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=upper_prices + lower_prices[::-1],
        fill='toself',
        fillcolor='rgba(68, 68, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence'
    ))
    
    # Predicted path
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=pred_prices,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=10),
        name='Predicted Price'
    ))
    
    # Today marker
    fig.add_trace(go.Scatter(
        x=["Today"],
        y=[last_close],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star'),
        name='Current Price'
    ))
    
    fig.update_layout(
        title=f'{ticker} Multi-Day Price Forecast',
        xaxis_title='Prediction Horizon',
        yaxis_title='Price ($)',
        height=500,
        showlegend=True
    )
    filepath = f"{OUTPUT_DIR}/{ticker}_forecast.html"
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_sentiment_gauge(sentiment_data: dict, ticker: str):
    """Plot sentiment gauge chart."""
    score = sentiment_data['sentiment_score']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{ticker} News Sentiment"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightgray"},
                {'range': [0.1, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=400)
    filepath = f"{OUTPUT_DIR}/{ticker}_sentiment.html"
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_residuals(y_test, predictions, model_name: str):
    """Plot residual distribution."""
    errors = predictions - y_test
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    sns.histplot(errors, kde=True, ax=axes[0])
    axes[0].set_title(f"Residual Distribution ({model_name})")
    axes[0].set_xlabel("Prediction Error")
    
    # Actual vs Predicted
    axes[1].scatter(y_test, predictions, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 'r--')
    axes[1].set_title(f"Actual vs Predicted ({model_name})")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    
    plt.tight_layout()
    filepath = f"{OUTPUT_DIR}/residuals.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("Stock Forecasting — Enhanced Model v2.0")
    print("Features: Sentiment + Multi-Day + Market Context + Classification + Backtest")
    print("=" * 70)
    
    ticker = input("\nEnter ticker symbol: ").strip().upper()
    
    # Download data
    df = download_stock_data(ticker)
    df = add_technical_indicators(df)
    
    # Download market context
    print("\n" + "-" * 50)
    print("LOADING MARKET CONTEXT")
    print("-" * 50)
    market_data = download_market_context()
    
    # Sentiment Analysis
    print("\n" + "-" * 50)
    print("SENTIMENT ANALYSIS")
    print("-" * 50)
    
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_data = sentiment_analyzer.analyze_ticker(ticker)
    
    print(f"\nOverall Sentiment: {sentiment_data['sentiment_label']}")
    print(f"Sentiment Score: {sentiment_data['sentiment_score']:.3f} (-1 to +1)")
    print(f"Headlines Analyzed: {sentiment_data['headline_count']}")
    print(f"  Positive: {sentiment_data['positive_pct']:.1f}%")
    print(f"  Neutral:  {sentiment_data['neutral_pct']:.1f}%")
    print(f"  Negative: {sentiment_data['negative_pct']:.1f}%")
    
    if sentiment_data['headlines']:
        print("\nRecent Headlines:")
        for i, h in enumerate(sentiment_data['headlines'][:5], 1):
            label = h['sentiment_label']
            print(f"  {i}. [{label:8}] {h['title'][:60]}...")
    
    # Create all targets
    horizons = [1, 5, 10, 30]
    df = create_multi_horizon_targets(df, horizons)
    df = create_price_range_targets(df)
    df = create_direction_target(df)
    
    # Build features with market context
    df_feat, features = build_features(df, sentiment_data['sentiment_score'], market_data)
    
    # Remove rows with NaN targets
    target_cols = [f"Target_{h}" for h in horizons] + ["Target_High", "Target_Low", "Direction"]
    df_feat = df_feat.dropna(subset=target_cols)
    
    # Train/test split
    split = int(len(df_feat) * 0.8)
    X_train = df_feat[features].iloc[:split]
    X_test = df_feat[features].iloc[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    latest_scaled = scaler.transform(df_feat[features].iloc[-1].values.reshape(1, -1))
    
    # Train models for each horizon
    print("\n" + "-" * 50)
    print("TRAINING MODELS")
    print("-" * 50)
    
    all_results = {}
    multi_day_predictions = {}
    
    for h in horizons:
        print(f"\nTraining models for {h}-day horizon...")
        y_train = df_feat[f"Target_{h}"].iloc[:split]
        y_test = df_feat[f"Target_{h}"].iloc[split:]
        
        # Tabular models
        models = train_models_for_horizon(X_train_scaled, y_train, 
                                          X_test_scaled, y_test, h)
        
        # LSTM (only for 1-day for efficiency)
        lstm_model, lstm_scaler, lstm_results = None, None, None
        if h == 1 and TENSORFLOW_AVAILABLE:
            closes = df["Close"].values
            lstm_model, lstm_scaler, lstm_results = train_lstm_model(closes, h)
            if lstm_results:
                models["LSTM"] = {
                    "model": lstm_model,
                    "scaler": lstm_scaler,
                    "predictions": lstm_results["predictions"],
                    "MAE": lstm_results["MAE"],
                    "RMSE": lstm_results["RMSE"]
                }
        
        all_results[h] = models
        
        # Ensemble prediction
        recent_closes = df["Close"].values if h == 1 else None
        ensemble = make_ensemble_prediction(
            models, latest_scaled, 
            lstm_model, lstm_scaler, recent_closes
        )
        
        # Apply volatility cap
        recent_vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
        if np.isnan(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
        
        last_close = df["Close"].iloc[-1]
        capped_pred = apply_volatility_cap(ensemble["mean"], last_close, recent_vol)
        
        multi_day_predictions[h] = {
            "mean": capped_pred,
            "std": ensemble["std"],
            "raw_mean": ensemble["mean"]
        }
    
    # Price Range Prediction
    print("\nTraining price range models (High/Low)...")
    y_train_high = df_feat["Target_High"].iloc[:split]
    y_test_high = df_feat["Target_High"].iloc[split:]
    y_train_low = df_feat["Target_Low"].iloc[:split]
    y_test_low = df_feat["Target_Low"].iloc[split:]
    
    high_models = train_models_for_horizon(X_train_scaled, y_train_high, X_test_scaled, y_test_high, 1)
    low_models = train_models_for_horizon(X_train_scaled, y_train_low, X_test_scaled, y_test_low, 1)
    
    pred_high = np.mean([high_models[m]["model"].predict(latest_scaled)[0] for m in high_models])
    pred_low = np.mean([low_models[m]["model"].predict(latest_scaled)[0] for m in low_models])
    
    # Direction Classification
    print("\nTraining direction classifier (UP/DOWN/FLAT)...")
    y_train_dir = df_feat["Direction"].iloc[:split]
    y_test_dir = df_feat["Direction"].iloc[split:]
    
    classifier = DirectionClassifier()
    class_results = classifier.train(X_train, y_train_dir, X_test, y_test_dir)
    direction_pred = classifier.predict_proba(df_feat[features].iloc[-1].values.reshape(1, -1))
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    last_close = df["Close"].iloc[-1]
    print(f"\nCurrent Price: ${last_close:.2f}")
    print(f"Sentiment: {sentiment_data['sentiment_label']} ({sentiment_data['sentiment_score']:.3f})")
    
    # Market context summary
    if market_data:
        print("\nMarket Context:")
        if "sp500" in market_data:
            sp500_ret = market_data["sp500"]["ret"].iloc[-1] * 100
            print(f"  S&P 500 (last day): {sp500_ret:+.2f}%")
        if "vix" in market_data:
            vix_level = market_data["vix"]["close"].iloc[-1]
            print(f"  VIX Level: {vix_level:.1f}")
    
    # Multi-day predictions
    print("\n{:<12} {:>15} {:>15} {:>15}".format(
        "Horizon", "Predicted Δ", "Predicted Price", "Change %"))
    print("-" * 60)
    
    for h in horizons:
        pred = multi_day_predictions[h]
        pred_price = last_close + pred["mean"]
        pct_change = (pred["mean"] / last_close) * 100
        
        print("{:<12} {:>+15.2f} {:>15.2f} {:>+14.2f}%".format(
            f"{h}-day", pred["mean"], pred_price, pct_change))
    
    # Price range prediction
    print("\n" + "-" * 50)
    print("PRICE RANGE PREDICTION (Next Day)")
    print("-" * 50)
    print(f"  Predicted High: ${last_close + pred_high:.2f} ({pred_high:+.2f})")
    print(f"  Predicted Low:  ${last_close + pred_low:.2f} ({pred_low:+.2f})")
    print(f"  Predicted Range: ${pred_high - pred_low:.2f}")
    
    # Direction classification
    print("\n" + "-" * 50)
    print("DIRECTION CLASSIFICATION")
    print("-" * 50)
    print(f"Classifier Accuracy: {class_results['accuracy']*100:.1f}%")
    if direction_pred:
        pred_dir = "UP" if direction_pred["prediction"] == 1 else ("DOWN" if direction_pred["prediction"] == -1 else "FLAT")
        print(f"Next Day Prediction: {pred_dir}")
        print("Confidence:")
        for label, conf in direction_pred["confidence"].items():
            bar = "█" * int(conf // 5) + "░" * (20 - int(conf // 5))
            print(f"  {label:5}: {bar} {conf:.1f}%")
    
    # Model performance comparison
    print("\n" + "-" * 50)
    print("MODEL PERFORMANCE (1-day horizon)")
    print("-" * 50)
    print("{:<18} {:>10} {:>10}".format("Model", "MAE", "RMSE"))
    print("-" * 40)
    for name, res in all_results[1].items():
        print("{:<18} {:>10.3f} {:>10.3f}".format(name, res["MAE"], res["RMSE"]))
    
    # Signal strength
    typical_change = df["Close"].diff().abs().mean()
    pred_1d = abs(multi_day_predictions[1]["mean"])
    
    print("\n" + "-" * 50)
    print("SIGNAL ANALYSIS")
    print("-" * 50)
    print(f"Typical daily move: ${typical_change:.2f}")
    print(f"Predicted 1-day move: ${pred_1d:.2f}")
    
    if pred_1d < 0.5 * typical_change:
        signal = "WEAK"
    elif pred_1d < 1.5 * typical_change:
        signal = "MODERATE"
    else:
        signal = "STRONG"
    print(f"Signal Strength: {signal}")
    
    # Backtesting
    print("\n" + "-" * 50)
    print("BACKTESTING")
    print("-" * 50)
    
    backtest_engine = BacktestEngine(initial_capital=10000)
    test_predictions = all_results[1]["XGBoost"]["predictions"]
    backtest_results = backtest_engine.run_backtest(
        df_feat.iloc[split:], 
        test_predictions
    )
    
    print(f"Initial Capital: $10,000")
    print(f"Final Capital: ${backtest_results['final_capital']:.2f}")
    print(f"Total Return: {backtest_results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {backtest_results['win_rate_pct']:.1f}%")
    print(f"Total Trades: {backtest_results['total_trades']}")
    
    # Visualizations
    print("\n" + "-" * 50)
    print("GENERATING CHARTS")
    print("-" * 50)
    
    plot_sentiment_gauge(sentiment_data, ticker)
    plot_price_history(df, ticker)
    plot_multi_day_predictions(last_close, multi_day_predictions, ticker, 
                               df.index[-30:].tolist())
    backtest_engine.plot_equity_curve(ticker)
    
    # Residuals for best model (exclude LSTM due to different array length)
    tabular_models = [m for m in all_results[1].keys() if m != "LSTM"]
    if tabular_models:
        best_model = min(tabular_models, key=lambda m: all_results[1][m]["MAE"])
        y_test_1d = df_feat["Target_1"].iloc[split:]
        plot_residuals(y_test_1d.values, all_results[1][best_model]["predictions"], 
                       best_model)
    
    # Correlation heatmap with new indicators
    plt.figure(figsize=(14, 10))
    corr_cols = ["Close", "Volume", "rsi", "macd", "bb_width", "atr", "stoch_k", "obv_pct"]
    corr_data = df[[c for c in corr_cols if c in df.columns]].corr()
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (Extended)")
    plt.tight_layout()
    filepath = f"{OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("    Do not use these predictions for actual trading decisions.")


if __name__ == "__main__":
    main()
