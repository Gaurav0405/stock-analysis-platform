import matplotlib
matplotlib.use('Agg')  # Fix for Flask/Windows thread crashes
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
import main  # Import the analysis module
from config import OUTPUT_DIR

app = Flask(__name__, static_folder='client/dist', static_url_path='')
CORS(app)  # Enable CORS for development

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    try:
        # Run analysis and get results
        # We need to modify main.analyze_single_ticker to return data
        # For now, we will capture its return value
        result = main.analyze_single_ticker(ticker)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def history():
    files = os.listdir(OUTPUT_DIR)
    # Group files by ticker and date
    history_data = []
    # This is a simple implementation, can be improved
    # We look for _predictions.csv files as they mark a completed run
    for f in files:
        if f.endswith('_predictions.csv'):
            parts = f.split('_')
            if len(parts) >= 3:
                ticker = parts[0]
                date = parts[1]
                history_data.append({
                    "ticker": ticker,
                    "date": date,
                    "files": [x for x in files if x.startswith(f"{ticker}_{date}")]
                })
    return jsonify(history_data)

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory(OUTPUT_DIR, filename)

def analyze_csv_file(filepath: str, ticker_name: str = "CUSTOM") -> dict:
    """Analyze uploaded CSV file through the full pipeline."""
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['Date', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain at least: {required_cols}")
        
        # Parse dates and set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing OHLV with Close if not present
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000  # Default volume
        
        # Run the same analysis as ticker-based
        from analysis.technical import add_technical_indicators, calculate_support_resistance
        from analysis.sentiment import SentimentAnalyzer
        from analysis.monte_carlo import MonteCarloSimulator
        from data.features import build_features, create_multi_horizon_targets
        from models.regression import train_models_for_horizon, make_ensemble_prediction, apply_volatility_cap
        from models.backtest import BacktestEngine
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        df = add_technical_indicators(df)
        
        # Sentiment (use neutral for custom data)
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_data = {
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'headline_count': 0,
            'positive_pct': 0,
            'neutral_pct': 100,
            'negative_pct': 0
        }
        
        # Support/Resistance & Patterns
        from analysis.technical import detect_patterns, detect_peaks_troughs, detect_trend
        
        # Calculate patterns on Close price
        closes = df['Close'].values
        peaks, troughs = detect_peaks_troughs(closes)
        patterns = detect_patterns(closes, peaks, troughs)
        trend = detect_trend(closes)
        
        sr_levels = calculate_support_resistance(df)
        
        # Monte Carlo
        mc = MonteCarloSimulator(n_simulations=1000, n_days=30)
        mc_stats = mc.run_simulation(df)
        
        # Create targets and features
        horizons = [1, 5, 10, 30]
        df = create_multi_horizon_targets(df, horizons)
        
        market_data = None  # No market context for custom CSV
        df_feat, features = build_features(df, sentiment_data['sentiment_score'], market_data)
        
        target_cols = [f"Target_{h}" for h in horizons]
        df_feat = df_feat.dropna(subset=target_cols)
        
        # Drop any remaining NaN values in features
        df_feat = df_feat.dropna()
        
        if len(df_feat) < 20:
            raise ValueError("Not enough valid data points after removing missing values. Need at least 20 rows.")

        
        # Train/test split
        split = int(len(df_feat) * 0.8)
        X_train = df_feat[features].iloc[:split]
        X_test = df_feat[features].iloc[split:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        latest_scaled = scaler.transform(df_feat[features].iloc[-1].values.reshape(1, -1))
        
        # Train models
        multi_day_predictions = {}
        all_results = {}
        
        for h in horizons:
            y_train = df_feat[f"Target_{h}"].iloc[:split]
            y_test = df_feat[f"Target_{h}"].iloc[split:]
            
            models = train_models_for_horizon(X_train_scaled, y_train, X_test_scaled, y_test, h)
            all_results[h] = models
            
            current_adx = df["adx"].iloc[-1] if "adx" in df.columns else 25
            ensemble = make_ensemble_prediction(models, latest_scaled, None, None, None, adx_score=current_adx)
            
            recent_vol = df["Close"].pct_change().rolling(20).std().iloc[-1]
            if np.isnan(recent_vol) or recent_vol <= 0:
                recent_vol = 0.02
            
            last_close = df["Close"].iloc[-1]
            capped_pred = apply_volatility_cap(ensemble["mean"], last_close, recent_vol)
            
            multi_day_predictions[h] = {
                "mean": capped_pred, "std": ensemble["std"], "raw_mean": ensemble["mean"]
            }
        
        # Backtest
        backtest_engine = BacktestEngine()
        test_predictions = all_results[1]["XGBoost"]["predictions"]
        backtest_results = backtest_engine.run_backtest(df_feat.iloc[split:], test_predictions)
        
        # Generate charts
        from visualization.charts import (plot_price_history, plot_multi_day_predictions,
                                         plot_monte_carlo, plot_support_resistance)
        
        plot_price_history(df, ticker_name)
        plot_multi_day_predictions(last_close, multi_day_predictions, ticker_name, df.index[-30:].tolist())
        plot_monte_carlo(mc, ticker_name)
        plot_support_resistance(df, sr_levels, ticker_name)
        backtest_engine.plot_equity_curve(ticker_name)
        
        # Export results
        from utils.export import export_predictions_to_csv
        prediction_csv = export_predictions_to_csv(ticker_name, multi_day_predictions, 
                                                   sentiment_data, backtest_results, {})
        
        # Build response
        last_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1].strftime("%Y-%m-%d")
        
        pred_5d = multi_day_predictions.get(5, {}).get("mean", 0)
        pred_price = last_price + pred_5d
        change_pct = (pred_price - last_price) / last_price * 100
        
        from config import get_output_path
        
        return {
            "ticker": ticker_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "last_price": last_price,
            "last_date": last_date,
            "sentiment": sentiment_data,
            "prediction_5d": {
                "price": float(pred_price),
                "change_pct": float(change_pct),
                "volatility": float(multi_day_predictions.get(5, {}).get("std", 0)),
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
                "resistance_levels": sr_levels.get("detected_resistances", []),
                "rsi": float(df["rsi"].iloc[-1]) if "rsi" in df.columns and not pd.isna(df["rsi"].iloc[-1]) else None,
                "macd": float(df["macd"].iloc[-1]) if "macd" in df.columns and not pd.isna(df["macd"].iloc[-1]) else None,
                "macd_signal": float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns and not pd.isna(df["macd_signal"].iloc[-1]) else None,
                "bb_upper": float(df["bb_high"].iloc[-1]) if "bb_high" in df.columns and not pd.isna(df["bb_high"].iloc[-1]) else None,
                "bb_lower": float(df["bb_low"].iloc[-1]) if "bb_low" in df.columns and not pd.isna(df["bb_low"].iloc[-1]) else None,
                "bb_mid": float(df["bb_mid"].iloc[-1]) if "bb_mid" in df.columns and not pd.isna(df["bb_mid"].iloc[-1]) else None,
                "atr": float(df["atr"].iloc[-1]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else None,
                "stoch_k": float(df["stoch_k"].iloc[-1]) if "stoch_k" in df.columns and not pd.isna(df["stoch_k"].iloc[-1]) else None,
                "stoch_d": float(df["stoch_d"].iloc[-1]) if "stoch_d" in df.columns and not pd.isna(df["stoch_d"].iloc[-1]) else None,
                "adx": float(df["adx"].iloc[-1]) if "adx" in df.columns and not pd.isna(df["adx"].iloc[-1]) else None
            },
            "files": {
                "prediction_csv": prediction_csv,
                "forecast_chart": get_output_path(ticker_name, "forecast"),
                "price_history_chart": get_output_path(ticker_name, "price_history"),
                "monte_carlo_chart": get_output_path(ticker_name, "monte_carlo"),
                "support_resistance_chart": get_output_path(ticker_name, "support_resistance"),
                "backtest_chart": get_output_path(ticker_name, "backtest"),
            }
        }
        
    except Exception as e:
        raise Exception(f"CSV analysis failed: {str(e)}")


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Extract ticker name from filename (remove .csv extension)
        ticker_name = os.path.splitext(filename)[0].upper()
        
        try:
            # Analyze the uploaded CSV
            results = analyze_csv_file(filepath, ticker_name)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Import image analysis module
            from analysis.image_analysis import analyze_chart_image
            import numpy as np
            
            # Run analysis to extract price line
            img_results = analyze_chart_image(filepath)
            prices = np.array(img_results['prices'])
            
            # Generate synthetic OHLV data to allow full technical analysis
            # We map the extracted line to 'Close' prices
            # And realistically derive Open/High/Low to create proper candlesticks
            # Open[t] should be close to Close[t-1] to minimize artificial gaps
            np.random.seed(42)  # For reproducibility
            
            opens = []
            highs = []
            lows = []
            volumes = []
            
            for i, price in enumerate(prices):
                # Determine previous close for realistic Open
                if i == 0:
                    prev_close = price # First day starts flat
                else:
                    prev_close = prices[i-1]
                
                # Open is previous close +/- small overnight random noise (0.2%)
                # This ensures the price moves happen 'intraday' (Open -> Close)
                # rather than 'overnight' (Close -> Open)
                open_p = prev_close * (1 + np.random.normal(0, 0.002))
                
                # Calculate intraday volatility based on the move size
                body_size = abs(price - open_p)
                min_volatility = price * 0.005 # Minimum 0.5% volatility
                day_volatility = max(min_volatility, body_size * 0.5)
                
                # High must be >= max(Open, Close)
                high_p = max(price, open_p) + abs(np.random.normal(0, day_volatility))
                
                # Low must be <= min(Open, Close)
                low_p = min(price, open_p) - abs(np.random.normal(0, day_volatility))
                
                volume = int(np.random.uniform(1000000, 5000000))
                
                opens.append(round(open_p, 2))
                highs.append(round(high_p, 2))
                lows.append(round(low_p, 2))
                volumes.append(volume)


            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now(), periods=len(prices), freq='D'),
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': volumes
            })
            
            # Save to temporary CSV to reuse the powerful analyze_csv_file pipeline
            temp_csv_name = f"extracted_{filename}.csv"
            temp_csv_path = os.path.join(UPLOAD_FOLDER, temp_csv_name)
            df.to_csv(temp_csv_path, index=False)
            
            # Reuse the full CSV analysis pipeline
            # This generates ALL charts: Forecast, Monte Carlo, Backtest, etc.
            results = analyze_csv_file(temp_csv_path, "CHART_EXTRACT")
            
            # Add the original image path to the results so the frontend can display it if needed
            # (though currently frontend doesn't use it, it's good metadata)
            results['files']['original_image'] = f"/uploads/{filename}"
            
            # Merge image-specific analysis (like patterns detected visually) if needed
            # For now, the CSV analysis is comprehensive enough
            
            return jsonify(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Image analysis failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

