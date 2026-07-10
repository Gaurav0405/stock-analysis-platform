"""
Quick Analysis — Fast chart generation without ML training
"""
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)


def quick_analyze(ticker: str) -> bool:
    """Generate basic charts quickly without ML training."""
    try:
        print(f"Quick analysis for {ticker}...")
        
        # Download data (minimal)
        print("  Downloading data...")
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            print(f"  No data for {ticker}")
            return False
        
        df.reset_index(inplace=True)
        
        # Calculate basic indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Returns'] = df['Close'].pct_change() * 100
        
        # Generate charts
        print("  Generating charts...")
        
        # 1. Price Chart with SMAs
        generate_price_chart(df, ticker)
        
        # 2. Volume Chart
        generate_volume_chart(df, ticker)
        
        # 3. RSI Chart
        generate_rsi_chart(df, ticker)
        
        # 4. Returns Distribution
        generate_returns_chart(df, ticker)
        
        # 5. Summary Dashboard
        generate_summary_chart(df, ticker)
        
        print(f"  Done! Charts saved to {CHARTS_DIR}")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def generate_price_chart(df, ticker):
    """Generate candlestick chart with SMAs."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'],
                             name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'],
                             name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    
    colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
                         marker_color=colors), row=2, col=1)
    
    fig.update_layout(
        title=f'{ticker} Price Analysis',
        template='plotly_dark',
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.write_html(f"{CHARTS_DIR}/{ticker}_price_history.html")


def generate_volume_chart(df, ticker):
    """Generate volume analysis chart."""
    fig = go.Figure()
    
    colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df['Close'], df['Open'])]
    
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors))
    
    avg_vol = df['Volume'].mean()
    fig.add_hline(y=avg_vol, line_dash="dash", line_color="yellow",
                  annotation_text=f"Avg: {avg_vol/1e6:.1f}M")
    
    fig.update_layout(
        title=f'{ticker} Volume Analysis',
        template='plotly_dark',
        height=400,
        yaxis_title='Volume'
    )
    
    fig.write_html(f"{CHARTS_DIR}/{ticker}_volume.html")


def generate_rsi_chart(df, ticker):
    """Generate RSI chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'],
                             mode='lines', name='RSI',
                             line=dict(color='#a855f7', width=2)))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1)
    
    current_rsi = df['RSI'].iloc[-1]
    status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    
    fig.update_layout(
        title=f'{ticker} RSI Indicator — Current: {current_rsi:.1f} ({status})',
        template='plotly_dark',
        height=350,
        yaxis=dict(range=[0, 100], title='RSI')
    )
    
    fig.write_html(f"{CHARTS_DIR}/{ticker}_rsi.html")


def generate_returns_chart(df, ticker):
    """Generate returns distribution."""
    returns = df['Returns'].dropna()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Daily Returns', 'Distribution'])
    
    colors = ['#22c55e' if r >= 0 else '#ef4444' for r in returns]
    fig.add_trace(go.Bar(x=df['Date'][1:], y=returns, marker_color=colors), row=1, col=1)
    
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color='#3b82f6'), row=1, col=2)
    
    fig.update_layout(
        title=f'{ticker} Returns Analysis — Avg: {returns.mean():.2f}%, Std: {returns.std():.2f}%',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    fig.write_html(f"{CHARTS_DIR}/{ticker}_returns.html")


def generate_summary_chart(df, ticker):
    """Generate summary dashboard."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=last['Close'],
        delta={'reference': prev['Close'], 'relative': True, 'valueformat': '.2%'},
        title={'text': f"{ticker} Current Price"},
        domain={'x': [0, 0.5], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=df['RSI'].iloc[-1],
        title={'text': "RSI"},
        domain={'x': [0.5, 1], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=df['Volume'].iloc[-1] / 1e6,
        title={'text': "Volume (M)"},
        number={'suffix': 'M'},
        domain={'x': [0, 0.5], 'y': [0, 0.5]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=df['Returns'].std(),
        title={'text': "Volatility (%)"},
        number={'suffix': '%'},
        domain={'x': [0.5, 1], 'y': [0, 0.5]}
    ))
    
    fig.update_layout(
        title=f'{ticker} Quick Summary',
        template='plotly_dark',
        height=400
    )
    
    fig.write_html(f"{CHARTS_DIR}/{ticker}_summary.html")


if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    quick_analyze(ticker)
