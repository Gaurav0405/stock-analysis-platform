"""
Results Accuracy Verification and Rating Script
Tests the quality and correctness of analysis outputs
"""
import sys
import os
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("ANALYSIS ACCURACY VERIFICATION AND RATING")
print("=" * 70)

# Load sample data
print("\n[1] Loading Sample Data...")
df = pd.read_csv('sample_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(f"    Loaded {len(df)} rows of stock data")
print(f"    Date range: {df.index[0]} to {df.index[-1]}")
print(f"    Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# 1. Technical Indicators Accuracy
print("\n[2] Technical Indicators Accuracy...")
from analysis.technical import add_technical_indicators

df = add_technical_indicators(df)

# Verify RSI is in valid range
rsi_valid = df['rsi'].dropna()
rsi_accuracy = 100 if (rsi_valid.min() >= 0 and rsi_valid.max() <= 100) else 0
print(f"    RSI range: {rsi_valid.min():.1f} - {rsi_valid.max():.1f} (Valid: 0-100)")

# Verify Bollinger Bands logic (Upper > Mid > Lower)
bb_check = (df['bb_high'] > df['bb_mid']).all() and (df['bb_mid'] > df['bb_low']).all()
bb_accuracy = 100 if bb_check else 0
print(f"    Bollinger Bands: Upper > Mid > Lower = {bb_check}")

# MACD should have reasonable values
macd_reasonable = df['macd'].dropna().std() < df['Close'].std() * 0.1
macd_accuracy = 100 if macd_reasonable else 50
print(f"    MACD volatility reasonable: {macd_reasonable}")

# 2. Monte Carlo Accuracy
print("\n[3] Monte Carlo Simulation Accuracy...")
from analysis.monte_carlo import MonteCarloSimulator

mc = MonteCarloSimulator(n_simulations=1000, n_days=30)
mc_stats = mc.run_simulation(df)

# max_loss should be negative (potential loss)
var_correct = mc_stats['max_loss'] <= 0
mc_accuracy = 100 if var_correct else 50
print(f"    Max Loss (VaR-like): {mc_stats['max_loss']:.2f}% (Should be negative)")
print(f"    Expected Return: {mc_stats['expected_return']:.2f}%")
print(f"    Prob Up: {mc_stats['prob_up']:.1f}%, Prob Down: {mc_stats['prob_down']:.1f}%")

# 3. Pattern Detection Accuracy
print("\n[4] Pattern Detection Accuracy...")
from analysis.technical import detect_patterns, detect_peaks_troughs, detect_trend

closes = df['Close'].values
peaks, troughs = detect_peaks_troughs(closes)
patterns = detect_patterns(closes, peaks, troughs)
trend = detect_trend(closes)

# Manual verification of trend
actual_trend = "Up" if closes[-1] > closes[0] else "Down"
trend_correct = (trend == actual_trend)
pattern_accuracy = 100 if trend_correct else 50
print(f"    Detected Trend: {trend}")
print(f"    Actual Trend (start vs end): {actual_trend}")
print(f"    Trend Detection Correct: {trend_correct}")
print(f"    Patterns Found: {patterns if patterns else 'None'}")
print(f"    Peaks Detected: {len(peaks)}, Troughs Detected: {len(troughs)}")

# 4. Model Prediction Accuracy (Backtest)
print("\n[5] ML Model Prediction Accuracy...")
from data.features import build_features, create_multi_horizon_targets
from models.regression import train_models_for_horizon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Prepare data
horizons = [1, 5]
df = create_multi_horizon_targets(df, horizons)

sentiment_data = {'sentiment_score': 0.1}
df_feat, features = build_features(df, sentiment_data['sentiment_score'], None)

target_cols = [f"Target_{h}" for h in horizons]
df_feat = df_feat.dropna(subset=target_cols)
df_feat = df_feat.dropna()

# Train/test split
split = int(len(df_feat) * 0.8)
X_train = df_feat[features].iloc[:split]
X_test = df_feat[features].iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = df_feat['Target_1'].iloc[:split]
y_test = df_feat['Target_1'].iloc[split:]

models = train_models_for_horizon(X_train_scaled, y_train, X_test_scaled, y_test, 1)

print(f"    Models trained: {list(models.keys())}")

# Calculate accuracy metrics for each model
model_scores = {}
for name, data in models.items():
    if 'predictions' in data:
        mae = mean_absolute_error(y_test, data['predictions'])
        # Calculate directional accuracy (predicts direction correctly)
        actual_direction = np.sign(y_test.values)
        pred_direction = np.sign(data['predictions'])
        directional_acc = (actual_direction == pred_direction).mean() * 100
        model_scores[name] = {'MAE': mae, 'Directional': directional_acc}
        print(f"    {name}: MAE=${mae:.2f}, Directional Accuracy={directional_acc:.1f}%")

# Average directional accuracy
avg_directional = np.mean([s['Directional'] for s in model_scores.values()])
model_accuracy = avg_directional

# 5. Support/Resistance Accuracy
print("\n[6] Support/Resistance Accuracy...")
from analysis.technical import calculate_support_resistance

sr = calculate_support_resistance(df)
current_price = sr['current_price']
s1, r1 = sr['support_1'], sr['resistance_1']

sr_reasonable = (s1 < current_price < r1)
sr_accuracy = 100 if sr_reasonable else 70
print(f"    Current Price: ${current_price:.2f}")
print(f"    Support 1: ${s1:.2f}, Resistance 1: ${r1:.2f}")
print(f"    Price between S1/R1: {sr_reasonable}")

# Calculate Overall Rating
print("\n" + "=" * 70)
print("ACCURACY RATING SUMMARY")
print("=" * 70)

scores = {
    'Technical Indicators (RSI, BB, MACD)': (rsi_accuracy + bb_accuracy + macd_accuracy) / 3,
    'Monte Carlo Simulation': mc_accuracy,
    'Trend/Pattern Detection': pattern_accuracy,
    'ML Model Predictions': model_accuracy,
    'Support/Resistance': sr_accuracy
}

for component, score in scores.items():
    bar = '#' * int(score / 5) + '-' * (20 - int(score / 5))
    print(f"  {component:40} [{bar}] {score:.0f}%")

overall_score = np.mean(list(scores.values()))
print(f"\n{'=' * 70}")
print(f"OVERALL ACCURACY RATING: {overall_score:.1f}/100")

if overall_score >= 80:
    grade = "A - Excellent"
elif overall_score >= 70:
    grade = "B - Good"
elif overall_score >= 60:
    grade = "C - Acceptable"
else:
    grade = "D - Needs Improvement"

print(f"GRADE: {grade}")
print("=" * 70)

# Recommendations
print("\nRECOMMENDATIONS:")
if model_accuracy < 60:
    print("  - Model directional accuracy is low. Consider more training data.")
if not trend_correct:
    print("  - Trend detection mismatch. May need parameter tuning.")
if overall_score >= 70:
    print("  - System is performing well for technical analysis use cases.")
    print("  - Predictions should be used as one input, not sole decision maker.")
