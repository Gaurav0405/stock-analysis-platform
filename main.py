"""
Stock Forecaster v3.0 — Modular Architecture
=============================================
Features:
- Sentiment Analysis (VADER + RSS)
- Multi-Day Predictions (1, 5, 10, 30 days)
- Market Context (S&P 500, VIX)
- Technical Indicators (RSI, MACD, BB, ATR, Stochastic, OBV)
- Support/Resistance Levels
- Direction Classification (UP/DOWN/FLAT)
- Monte Carlo Simulation
- Backtesting Engine
- CSV Export
- Multi-Ticker Mode
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Local modules
from config import OUTPUT_DIR, DEFAULT_HORIZONS, MC_SIMULATIONS, MC_DAYS, get_output_path
from data.downloader import download_stock_data, download_market_context, get_fundamental_data
from data.features import (build_features, create_multi_horizon_targets,
                           create_price_range_targets, create_direction_target)
from analysis.technical import (add_technical_indicators, calculate_support_resistance,
                                detect_patterns, detect_trend, detect_peaks_troughs)
from analysis.sentiment import SentimentAnalyzer
from analysis.monte_carlo import MonteCarloSimulator
from models.regression import (train_models_for_horizon, train_lstm_model,
                               make_ensemble_prediction, apply_volatility_cap)
from models.classification import DirectionClassifier
from models.backtest import BacktestEngine
from visualization.charts import (plot_price_history, plot_multi_day_predictions,
                                   plot_sentiment_gauge, plot_residuals,
                                   plot_monte_carlo, plot_correlation_heatmap,
                                   plot_volume_analysis, plot_returns_distribution,
                                   plot_feature_importance, plot_support_resistance,
                                   plot_model_comparison, plot_drawdown,
                                   plot_prediction_accuracy)
from utils.export import export_predictions_to_csv, export_backtest_trades, run_multi_ticker_analysis


def analyze_single_ticker(ticker: str) -> dict:
    """Run full analysis on a single ticker."""
    
    # Download data
    df = download_stock_data(ticker)
    df = add_technical_indicators(df)
    
    # Fundamental data
    print("\n" + "-" * 50)
    print("FUNDAMENTAL DATA")
    print("-" * 50)
    fundamentals = get_fundamental_data(ticker)
    if fundamentals:
        print(f"  Company: {fundamentals.get('company_name', ticker)}")
        print(f"  Sector: {fundamentals.get('sector', 'N/A')}")
        if fundamentals.get('pe_ratio'):
            print(f"  P/E Ratio: {fundamentals['pe_ratio']:.2f}")
        if fundamentals.get('eps'):
            print(f"  EPS: ${fundamentals['eps']:.2f}")
        if fundamentals.get('market_cap'):
            print(f"  Market Cap: ${fundamentals['market_cap']:,.0f}")
    
    # Market context
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
    print(f"Sentiment Score: {sentiment_data['sentiment_score']:.3f}")
    print(f"Headlines Analyzed: {sentiment_data['headline_count']}")
    print(f"  Positive: {sentiment_data['positive_pct']:.1f}%")
    print(f"  Neutral:  {sentiment_data['neutral_pct']:.1f}%")
    print(f"  Negative: {sentiment_data['negative_pct']:.1f}%")
    
    # Support/Resistance
    print("\n" + "-" * 50)
    print("SUPPORT & RESISTANCE LEVELS")
    print("-" * 50)
    sr_levels = calculate_support_resistance(df)
    print(f"  Current Price: ${sr_levels['current_price']:.2f}")
    print(f"  Pivot: ${sr_levels['pivot']:.2f}")
    print(f"  Resistance 1: ${sr_levels['resistance_1']:.2f}")
    print(f"  Resistance 2: ${sr_levels['resistance_2']:.2f}")
    print(f"  Support 1: ${sr_levels['support_1']:.2f}")
    print(f"  Support 2: ${sr_levels['support_2']:.2f}")
    
    # Patterns & Trend
    print("\n" + "-" * 50)
    print("TECHNICAL SIGNALS")
    print("-" * 50)
    closes = df["Close"].values
    trend = detect_trend(closes)
    peaks, troughs = detect_peaks_troughs(closes)
    patterns = detect_patterns(closes, peaks, troughs)
    
    print(f"  Trend: {trend}")
    if patterns:
        print(f"  Patterns: {', '.join(patterns)}")
    else:
        print("  Patterns: None detected")
    
    # Monte Carlo Simulation
    print("\n" + "-" * 50)
    print("MONTE CARLO SIMULATION")
    print("-" * 50)
    mc = MonteCarloSimulator(n_simulations=MC_SIMULATIONS, n_days=MC_DAYS)
    mc_stats = mc.run_simulation(df)
    print(f"  Simulations: {MC_SIMULATIONS}")
    print(f"  Days Forward: {MC_DAYS}")
    print(f"  Expected Price: ${mc_stats['mean_final']:.2f}")
    print(f"  95% Range: ${mc_stats['percentile_5']:.2f} - ${mc_stats['percentile_95']:.2f}")
    print(f"  Probability UP: {mc_stats['prob_up']:.1f}%")
    print(f"  Expected Return: {mc_stats['expected_return']:+.2f}%")
    
    # Create targets
    horizons = DEFAULT_HORIZONS
    df = create_multi_horizon_targets(df, horizons)
    df = create_price_range_targets(df)
    df = create_direction_target(df)
    
    # Build features
    df_feat, features = build_features(df, sentiment_data['sentiment_score'], market_data)
    
    target_cols = [f"Target_{h}" for h in horizons] + ["Target_High", "Target_Low", "Direction"]
    df_feat = df_feat.dropna(subset=target_cols)
    
    # Train/test split
    split = int(len(df_feat) * 0.8)
    X_train = df_feat[features].iloc[:split]
    X_test = df_feat[features].iloc[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    latest_scaled = scaler.transform(df_feat[features].iloc[-1].values.reshape(1, -1))
    
    # Train models
    print("\n" + "-" * 50)
    print("TRAINING MODELS")
    print("-" * 50)
    
    all_results = {}
    multi_day_predictions = {}
    
    for h in horizons:
        print(f"\nTraining {h}-day models...")
        y_train = df_feat[f"Target_{h}"].iloc[:split]
        y_test = df_feat[f"Target_{h}"].iloc[split:]
        
        models = train_models_for_horizon(X_train_scaled, y_train, X_test_scaled, y_test, h)
        
        # LSTM removed for stability as per user request
        lstm_model, lstm_scaler, lstm_results = None, None, None
        # if h == 1:
        #     closes = df["Close"].values
        #     lstm_model, lstm_scaler, lstm_results = train_lstm_model(closes, h)
        #     if lstm_results:
        #         models["LSTM"] = {
        #             "model": lstm_model, "scaler": lstm_scaler,
        #             "predictions": lstm_results["predictions"],
        #             "MAE": lstm_results["MAE"], "RMSE": lstm_results["RMSE"]
        #         }
        
        all_results[h] = models
        
        all_results[h] = models
        
        recent_closes = df["Close"].values if h == 1 else None
        
        # Get latest ADX for dynamic weighting
        current_adx = df["adx"].iloc[-1]
        print(f"DEBUG: Horizon {h} - Current ADX: {current_adx:.2f}")
        
        ensemble = make_ensemble_prediction(models, latest_scaled, lstm_model, 
                                          lstm_scaler, recent_closes, adx_score=current_adx)
        
        recent_vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
        if np.isnan(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
        
        last_close = df["Close"].iloc[-1]
        capped_pred = apply_volatility_cap(ensemble["mean"], last_close, recent_vol)
        
        multi_day_predictions[h] = {
            "mean": capped_pred, "std": ensemble["std"], "raw_mean": ensemble["mean"]
        }
    
    # Price Range
    print("\nTraining price range models...")
    y_train_high = df_feat["Target_High"].iloc[:split]
    y_test_high = df_feat["Target_High"].iloc[split:]
    y_train_low = df_feat["Target_Low"].iloc[:split]
    y_test_low = df_feat["Target_Low"].iloc[split:]
    
    high_models = train_models_for_horizon(X_train_scaled, y_train_high, X_test_scaled, y_test_high, 1)
    low_models = train_models_for_horizon(X_train_scaled, y_train_low, X_test_scaled, y_test_low, 1)
    
    pred_high = np.mean([high_models[m]["model"].predict(latest_scaled)[0] for m in high_models])
    pred_low = np.mean([low_models[m]["model"].predict(latest_scaled)[0] for m in low_models])
    
    # Direction Classification
    print("\nTraining direction classifier...")
    y_train_dir = df_feat["Direction"].iloc[:split]
    y_test_dir = df_feat["Direction"].iloc[split:]
    
    classifier = DirectionClassifier()
    class_results = classifier.train(X_train, y_train_dir, X_test, y_test_dir)
    direction_pred = classifier.predict_proba(df_feat[features].iloc[-1].values.reshape(1, -1))
    
    # Results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    last_close = df["Close"].iloc[-1]
    print(f"\nCurrent Price: ${last_close:.2f}")
    print(f"Sentiment: {sentiment_data['sentiment_label']} ({sentiment_data['sentiment_score']:.3f})")
    
    if market_data:
        print("\nMarket Context:")
        if "sp500" in market_data:
            sp500_ret = market_data["sp500"]["ret"].iloc[-1] * 100
            print(f"  S&P 500 (last day): {sp500_ret:+.2f}%")
        if "vix" in market_data:
            vix_level = market_data["vix"]["close"].iloc[-1]
            print(f"  VIX Level: {vix_level:.1f}")
    
    print("\n{:<12} {:>15} {:>15} {:>15}".format("Horizon", "Predicted Δ", "Price", "Change %"))
    print("-" * 60)
    for h in horizons:
        pred = multi_day_predictions[h]
        pred_price = last_close + pred["mean"]
        pct_change = (pred["mean"] / last_close) * 100
        print("{:<12} {:>+15.2f} {:>15.2f} {:>+14.2f}%".format(f"{h}-day", pred["mean"], pred_price, pct_change))
    
    print("\n" + "-" * 50)
    print("PRICE RANGE PREDICTION (Next Day)")
    print("-" * 50)
    print(f"  Predicted High: ${last_close + pred_high:.2f} ({pred_high:+.2f})")
    print(f"  Predicted Low:  ${last_close + pred_low:.2f} ({pred_low:+.2f})")
    
    print("\n" + "-" * 50)
    print("DIRECTION CLASSIFICATION")
    print("-" * 50)
    print(f"Accuracy: {class_results['accuracy']*100:.1f}%")
    if direction_pred:
        pred_dir = "UP" if direction_pred["prediction"] == 1 else ("DOWN" if direction_pred["prediction"] == -1 else "FLAT")
        print(f"Prediction: {pred_dir}")
        for label, conf in direction_pred["confidence"].items():
            bar = "█" * int(conf // 5) + "░" * (20 - int(conf // 5))
            print(f"  {label:5}: {bar} {conf:.1f}%")
    
    # Backtesting
    print("\n" + "-" * 50)
    print("BACKTESTING")
    print("-" * 50)
    
    backtest_engine = BacktestEngine()
    test_predictions = all_results[1]["XGBoost"]["predictions"]
    backtest_results = backtest_engine.run_backtest(df_feat.iloc[split:], test_predictions)
    
    print(f"Initial Capital: $10,000")
    print(f"Final Capital: ${backtest_results['final_capital']:.2f}")
    print(f"Total Return: {backtest_results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {backtest_results['win_rate_pct']:.1f}%")
    
    # Charts
    print("\n" + "-" * 50)
    print("GENERATING CHARTS")
    print("-" * 50)
    
    plot_sentiment_gauge(sentiment_data, ticker)
    plot_price_history(df, ticker)
    plot_multi_day_predictions(last_close, multi_day_predictions, ticker, df.index[-30:].tolist())
    plot_monte_carlo(mc, ticker)
    backtest_engine.plot_equity_curve(ticker)
    
    tabular_models = [m for m in all_results[1].keys() if m != "LSTM"]
    if tabular_models:
        best_model = min(tabular_models, key=lambda m: all_results[1][m]["MAE"])
        y_test_1d = df_feat["Target_1"].iloc[split:]
        plot_residuals(y_test_1d.values, all_results[1][best_model]["predictions"], best_model, ticker)
    
    plot_correlation_heatmap(df, ticker)
    
    # New charts
    plot_volume_analysis(df, ticker)
    plot_returns_distribution(df, ticker)
    plot_feature_importance(all_results[1], features, ticker)
    plot_support_resistance(df, sr_levels, ticker)
    plot_model_comparison(all_results, ticker)
    plot_drawdown(backtest_engine.equity_curve, ticker)
    
    # Prediction accuracy for all models
    model_preds = {m: all_results[1][m]["predictions"] for m in tabular_models}
    plot_prediction_accuracy(y_test_1d.values, model_preds, ticker)
    
    # CSV Export
    print("\n" + "-" * 50)
    print("EXPORTING DATA")
    print("-" * 50)
    # Export predictions (includes sentiment and backtest)
    prediction_csv = export_predictions_to_csv(ticker, multi_day_predictions, 
                                             sentiment_data, backtest_results,
                                             fundamentals)
    
    # Export backtest trades
    export_backtest_trades(ticker, backtest_engine.trades)
    
    # Export Monte Carlo stats
    from utils.export import export_monte_carlo_results
    export_monte_carlo_results(ticker, mc_stats, mc.get_percentile_paths())

    print("\n" + "=" * 50)
    print(f"ANALYSIS COMPLETE: {ticker}")
    print("=" * 50)
    print("\n⚠️  DISCLAIMER: Educational purposes only. Not financial advice.")
    
    # Calculate metrics for API return
    last_price = df["Close"].iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    # Get 5-day prediction if available
    pred_5d = multi_day_predictions.get(5, {}).get("mean", 0)
    pred_price = last_price + pred_5d
    change_pct = (pred_price - last_price) / last_price * 100
    
    results = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "last_price": float(last_price),
        "last_date": last_date,
        "sentiment": {
            "score": sentiment_data.get("sentiment_score", 0),
            "label": sentiment_data.get("sentiment_label", "NEUTRAL")
        },
        "prediction_5d": {
            "price": float(pred_price),
            "change_pct": float(change_pct),
            "volatility": float(multi_day_predictions.get(5, {}).get("std", 0)),
            "adx_score": float(df["adx"].iloc[-1])
        },
        "backtest": {
            "return_pct": backtest_results.get("total_return_pct", 0),
            "sharpe": backtest_results.get("sharpe_ratio", 0),
            "win_rate": backtest_results.get("win_rate_pct", 0)
        },
        "technical_analysis": {
            "trend": trend,
            "patterns": patterns,
            "support_levels": sr_levels.get("detected_supports", []),
            "resistance_levels": sr_levels.get("detected_resistances", [])
        },
        "files": {
            "prediction_csv": prediction_csv,
            "forecast_chart": get_output_path(ticker, "forecast"),
            "sentiment_chart": get_output_path(ticker, "sentiment"),
            "backtest_chart": get_output_path(ticker, "backtest"),
            "monte_carlo_chart": get_output_path(ticker, "monte_carlo"),
            "price_history_chart": get_output_path(ticker, "price_history"),
            "volume_chart": get_output_path(ticker, "volume"),
            "returns_chart": get_output_path(ticker, "returns"),
            "importance_chart": get_output_path(ticker, "importance"),
            "support_resistance_chart": get_output_path(ticker, "support_resistance"),
            "model_comparison_chart": get_output_path(ticker, "model_comparison"),
            "drawdown_chart": get_output_path(ticker, "drawdown"),
            "accuracy_chart": get_output_path(ticker, "accuracy"),
            "correlation_chart": get_output_path(ticker, "correlation", "png"),
            "residuals_chart": get_output_path(ticker, "residuals", "png")
        }
    }
    
    return results


def main():
    print("=" * 70)
    print("Stock Forecaster v3.0 — Modular Architecture")
    print("=" * 70)
    
    mode = input("\nMode: [1] Single Ticker  [2] Multi-Ticker: ").strip()
    
    if mode == "2":
        tickers_input = input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        run_multi_ticker_analysis(tickers, analyze_single_ticker)
    else:
        ticker = input("Enter ticker symbol: ").strip().upper()
        analyze_single_ticker(ticker)


if __name__ == "__main__":
    main()
