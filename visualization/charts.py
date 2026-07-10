"""
Visualization Module
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import OUTPUT_DIR, get_output_path


def plot_price_history(df: pd.DataFrame, ticker: str):
    """Plot candlestick chart with technical indicators."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.02)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    
    if "bb_high" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"],
                                 line=dict(color='gray', dash='dash'),
                                 name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"],
                                 line=dict(color='gray', dash='dash'),
                                 name='BB Lower'), row=1, col=1)
    
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
                                 line=dict(color='purple'), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                                 line=dict(color='blue'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                                 line=dict(color='orange'), name='Signal'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False, xaxis_rangeslider_visible=False)
    filepath = get_output_path(ticker, "price_history")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_multi_day_predictions(last_close: float, predictions: dict,
                                ticker: str, dates: list):
    """Plot multi-day price forecasts with confidence intervals."""
    horizons = sorted(predictions.keys())
    pred_prices = [last_close + predictions[h]["mean"] for h in horizons]
    upper = [last_close + predictions[h]["mean"] + 1.96 * predictions[h]["std"] for h in horizons]
    lower = [last_close + predictions[h]["mean"] - 1.96 * predictions[h]["std"] for h in horizons]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0] + horizons, y=[last_close] + pred_prices,
                             mode='lines+markers', name='Prediction',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=horizons, y=upper, mode='lines', name='95% Upper',
                             line=dict(color='lightblue', dash='dash')))
    fig.add_trace(go.Scatter(x=horizons, y=lower, mode='lines', name='95% Lower',
                             line=dict(color='lightblue', dash='dash'),
                             fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)'))
    
    fig.update_layout(title=f'{ticker} Multi-Day Price Forecast',
                      xaxis_title='Prediction Horizon (days)',
                      yaxis_title='Price ($)', height=500, showlegend=True)
    filepath = get_output_path(ticker, "forecast")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_sentiment_gauge(sentiment_data: dict, ticker: str):
    """Plot sentiment gauge visualization."""
    score = sentiment_data['sentiment_score']
    label = sentiment_data['sentiment_label']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': f"{ticker} Sentiment - {label}"},
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
            'threshold': {'line': {'color': "black", 'width': 4},
                         'thickness': 0.75, 'value': score}
        }
    ))
    
    fig.update_layout(height=400)
    filepath = get_output_path(ticker, "sentiment")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_residuals(y_test, predictions, model_name: str, ticker: str = ""):
    """Plot residual distribution."""
    errors = predictions - y_test
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(errors, kde=True, ax=axes[0])
    axes[0].set_title(f"Residual Distribution ({model_name})")
    axes[0].set_xlabel("Prediction Error")
    
    axes[1].scatter(y_test, predictions, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_title(f"Actual vs Predicted ({model_name})")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    
    plt.tight_layout()
    filepath = get_output_path(ticker, "residuals", "png") if ticker else f"{OUTPUT_DIR}/residuals.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_monte_carlo(simulator, ticker: str):
    """Plot Monte Carlo simulation results."""
    paths = simulator.get_percentile_paths()
    if not paths:
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(y=paths["p5"], mode='lines', name='5th Percentile',
                             line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(y=paths["p95"], mode='lines', name='95th Percentile',
                             line=dict(color='green', dash='dash'),
                             fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)'))
    fig.add_trace(go.Scatter(y=paths["p25"], mode='lines', name='25th Percentile',
                             line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(y=paths["p75"], mode='lines', name='75th Percentile',
                             line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(y=paths["mean"], mode='lines', name='Mean',
                             line=dict(color='black', width=2)))
    
    fig.update_layout(title=f'{ticker} Monte Carlo Simulation ({simulator.n_simulations} paths, {simulator.n_days} days)',
                      xaxis_title='Days', yaxis_title='Price ($)', height=500)
    filepath = get_output_path(ticker, "monte_carlo")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_correlation_heatmap(df: pd.DataFrame, ticker: str = ""):
    """Plot feature correlation heatmap."""
    plt.figure(figsize=(14, 10))
    corr_cols = ["Close", "Volume", "rsi", "macd", "bb_width", "atr", "stoch_k", "obv_pct"]
    corr_data = df[[c for c in corr_cols if c in df.columns]].corr()
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    filepath = get_output_path(ticker, "correlation", "png") if ticker else f"{OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_volume_analysis(df: pd.DataFrame, ticker: str):
    """Plot volume analysis with price overlay."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Price with moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines',
                             name='Close', line=dict(color='blue', width=1)), row=1, col=1)
    
    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], mode='lines',
                                 name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    
    # Volume bars colored by price direction
    colors = ['green' if df["Close"].iloc[i] >= df["Open"].iloc[i] else 'red' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume',
                         marker_color=colors, opacity=0.7), row=2, col=1)
    
    # Volume moving average
    vol_ma = df["Volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, mode='lines',
                             name='Vol MA(20)', line=dict(color='purple', width=2)), row=2, col=1)
    
    fig.update_layout(title=f'{ticker} Volume Analysis', height=600, showlegend=True)
    filepath = get_output_path(ticker, "volume")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_returns_distribution(df: pd.DataFrame, ticker: str):
    """Plot daily returns distribution with statistics."""
    returns = df["Close"].pct_change().dropna() * 100
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Returns Distribution", "Returns Over Time"))
    
    # Histogram
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns',
                               marker_color='blue', opacity=0.7), row=1, col=1)
    
    # Add normal distribution overlay
    mean_ret = returns.mean()
    std_ret = returns.std()
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_ret) / std_ret) ** 2)
    normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50
    
    fig.add_trace(go.Scatter(x=x_range, y=normal_dist, mode='lines',
                             name='Normal Dist', line=dict(color='red', width=2)), row=1, col=1)
    
    # Returns over time
    fig.add_trace(go.Scatter(x=df.index[1:], y=returns, mode='lines',
                             name='Returns', line=dict(color='green', width=1)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Add annotations with stats
    stats_text = f"Mean: {mean_ret:.3f}%<br>Std: {std_ret:.3f}%<br>Skew: {returns.skew():.2f}<br>Kurt: {returns.kurtosis():.2f}"
    fig.add_annotation(x=0.15, y=0.95, xref="paper", yref="paper",
                       text=stats_text, showarrow=False, font=dict(size=12),
                       bgcolor="white", bordercolor="black", borderwidth=1)
    
    fig.update_layout(title=f'{ticker} Returns Analysis', height=500, showlegend=True)
    filepath = get_output_path(ticker, "returns")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_feature_importance(models: dict, features: list, ticker: str):
    """Plot feature importance from tree-based models."""
    importance_data = {}
    
    for name, model_info in models.items():
        model = model_info.get("model")
        if hasattr(model, 'feature_importances_'):
            importance_data[name] = model.feature_importances_
    
    if not importance_data:
        print("  No feature importance available (no tree-based models)")
        return
    
    # Use first available model
    model_name = list(importance_data.keys())[0]
    importances = importance_data[model_name]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    top_features = [features[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    fig = go.Figure(go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f'{ticker} Feature Importance ({model_name})',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )
    filepath = get_output_path(ticker, "importance")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_support_resistance(df: pd.DataFrame, sr_levels: dict, ticker: str):
    """Plot price chart with support and resistance levels."""
    fig = go.Figure()
    
    # Recent price data (last 60 days)
    recent_df = df.tail(60)
    
    fig.add_trace(go.Candlestick(
        x=recent_df.index, open=recent_df["Open"], high=recent_df["High"],
        low=recent_df["Low"], close=recent_df["Close"], name="Price"
    ))
    
    # Add horizontal lines for levels
    fig.add_hline(y=sr_levels['pivot'], line_dash="solid", line_color="blue",
                  annotation_text=f"Pivot: ${sr_levels['pivot']:.2f}")
    fig.add_hline(y=sr_levels['resistance_1'], line_dash="dash", line_color="red",
                  annotation_text=f"R1: ${sr_levels['resistance_1']:.2f}")
    fig.add_hline(y=sr_levels['resistance_2'], line_dash="dot", line_color="darkred",
                  annotation_text=f"R2: ${sr_levels['resistance_2']:.2f}")
    fig.add_hline(y=sr_levels['support_1'], line_dash="dash", line_color="green",
                  annotation_text=f"S1: ${sr_levels['support_1']:.2f}")
    fig.add_hline(y=sr_levels['support_2'], line_dash="dot", line_color="darkgreen",
                  annotation_text=f"S2: ${sr_levels['support_2']:.2f}")
    
    fig.update_layout(
        title=f'{ticker} Support & Resistance Levels',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        xaxis_rangeslider_visible=False
    )
    filepath = get_output_path(ticker, "support_resistance")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_model_comparison(all_results: dict, ticker: str):
    """Plot model performance comparison across horizons."""
    horizons = sorted(all_results.keys())
    models = list(all_results[horizons[0]].keys())
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE by Model", "RMSE by Model"))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, model in enumerate(models):
        mae_values = [all_results[h].get(model, {}).get("MAE", 0) for h in horizons]
        rmse_values = [all_results[h].get(model, {}).get("RMSE", 0) for h in horizons]
        
        fig.add_trace(go.Bar(name=model, x=[f"{h}d" for h in horizons], y=mae_values,
                             marker_color=colors[i % len(colors)]), row=1, col=1)
        fig.add_trace(go.Bar(name=model, x=[f"{h}d" for h in horizons], y=rmse_values,
                             marker_color=colors[i % len(colors)], showlegend=False), row=1, col=2)
    
    fig.update_layout(
        title=f'{ticker} Model Performance Comparison',
        barmode='group',
        height=500
    )
    filepath = get_output_path(ticker, "model_comparison")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_drawdown(equity_curve: list, ticker: str):
    """Plot drawdown analysis from backtest equity curve."""
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05,
                        subplot_titles=("Portfolio Value", "Drawdown %"))
    
    # Equity curve
    fig.add_trace(go.Scatter(y=equity, mode='lines', name='Portfolio',
                             line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=peak, mode='lines', name='Peak',
                             line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(y=-drawdown, mode='lines', name='Drawdown',
                             fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                             line=dict(color='red', width=1)), row=2, col=1)
    
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    fig.add_annotation(x=max_dd_idx, y=-max_dd, text=f"Max DD: {max_dd:.1f}%",
                       showarrow=True, arrowhead=2, row=2, col=1)
    
    fig.update_layout(title=f'{ticker} Drawdown Analysis', height=600, showlegend=True)
    filepath = get_output_path(ticker, "drawdown")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")


def plot_prediction_accuracy(y_test, predictions: dict, ticker: str):
    """Plot prediction accuracy scatter for multiple models."""
    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, 
                        subplot_titles=list(predictions.keys()))
    
    for idx, (model_name, preds) in enumerate(predictions.items()):
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Align lengths
        min_len = min(len(y_test), len(preds))
        y_actual = y_test[:min_len] if hasattr(y_test, '__getitem__') else y_test.values[:min_len]
        y_pred = preds[:min_len]
        
        fig.add_trace(go.Scatter(x=y_actual, y=y_pred, mode='markers',
                                 marker=dict(size=5, opacity=0.5),
                                 name=model_name), row=row, col=col)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 showlegend=False), row=row, col=col)
    
    fig.update_layout(title=f'{ticker} Prediction Accuracy by Model', height=400 * rows)
    filepath = get_output_path(ticker, "accuracy")
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")

