"""
Regression Models Module
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from config import RANDOM_STATE

# Minimal imports to reduce conflict risk
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Optional TensorFlow: REMOVED for stability
TENSORFLOW_AVAILABLE = False


def train_models_for_horizon(X_train, y_train, X_test, y_test, horizon: int) -> dict:
    """Train ensemble models (XGBoost, RF, Ridge, SVR, KNN)."""
    models = {}
    
    # 1. XGBoost (Trend)
    try:
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                           random_state=RANDOM_STATE, n_jobs=1, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        models["XGBoost"] = {
            "model": xgb,
            "predictions": xgb_pred,
            "MAE": mean_absolute_error(y_test, xgb_pred)
        }
    except Exception as e:
        print(f"XGBoost error: {e}")

    # 2. Random Forest (Stability)
    try:
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, 
                                   random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        models["RandomForest"] = {
            "model": rf,
            "predictions": rf_pred,
            "MAE": mean_absolute_error(y_test, rf_pred)
        }
    except Exception as e:
        print(f"RandomForest error: {e}")

    # 3. Ridge Regression (Baseline)
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        models["Linear"] = {
            "model": ridge,
            "predictions": ridge_pred,
            "MAE": mean_absolute_error(y_test, ridge_pred)
        }
    except Exception as e:
        print(f"Ridge error: {e}")
        
    # 4. SVR (Cycles)
    try:
        svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        svr.fit(X_train, y_train)
        svr_pred = svr.predict(X_test)
        models["SVR"] = {
            "model": svr,
            "predictions": svr_pred,
            "MAE": mean_absolute_error(y_test, svr_pred)
        }
    except Exception as e:
        print(f"SVR error: {e}")

    # 5. KNN (Pattern Matching)
    try:
        knn = KNeighborsRegressor(n_neighbors=20, weights='distance')
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        models["KNN"] = {
            "model": knn,
            "predictions": knn_pred,
            "MAE": mean_absolute_error(y_test, knn_pred)
        }
    except Exception as e:
        print(f"KNN error: {e}")
    
    return models


def train_lstm_model(closes: np.ndarray, horizon: int, window: int = 60):
    """Train LSTM model for sequence-based prediction."""
    # LSTM removed for stability
    return None, None, None


def make_ensemble_prediction(models: dict, X_latest, lstm_model=None, 
                             lstm_scaler=None, recent_closes=None, adx_score: float = None) -> dict:
    """Combine predictions from 'Council of 6' with Dynamic Weighting."""
    predictions = []
    weights = []
    
    # Base weights
    model_weights = {
        "XGBoost": 1.0,
        "RandomForest": 1.0,
        "Linear": 0.5,
        "SVR": 1.0,
        "KNN": 1.0,
        "LSTM": 1.0
    }
    
    # Dynamic Adjustment based on ADX (Trend Strength)
    if adx_score is not None:
        if adx_score < 25:
            # Sideways Market: TRUST Pattern Matchers, IGNORE Trend
            model_weights["SVR"] *= 2.0
            model_weights["KNN"] *= 2.0
            model_weights["XGBoost"] *= 0.1  # Was 0.5 - penalized heavily
            model_weights["RandomForest"] *= 0.1
            print(f"  [Dynamic Weighting] ADX {adx_score:.1f} < 25 (Sideways): Boosting SVR/KNN, Muting Trend")
        else:
            # Trending Market: TRUST Trend, IGNORE Patterns
            model_weights["XGBoost"] *= 2.0
            model_weights["RandomForest"] *= 1.5
            model_weights["SVR"] *= 0.1      # Was 0.5 - penalized heavily
            model_weights["KNN"] *= 0.1
            print(f"  [Dynamic Weighting] ADX {adx_score:.1f} >= 25 (Trend): Boosting XGB/RF, Muting SVR/KNN")

    # 1. Standard Models
    for name, res in models.items():
        if "model" in res:
            try:
                pred = res["model"].predict(X_latest)[0]
                if isinstance(pred, (np.ndarray, list)):
                    pred = pred[0]
                
                # Apply weight
                w = model_weights.get(name, 1.0)
                predictions.append(float(pred))
                weights.append(w)
                print(f"  Model {name}: {pred:.2f} (Weight: {w:.1f})")
                
            except Exception as e:
                print(f"Predict error {name}: {e}")

    # 2. LSTM
    if lstm_model and lstm_scaler and recent_closes is not None:
        try:
            window = 60
            scaled = lstm_scaler.transform(recent_closes[-window:].reshape(-1, 1))
            lstm_input = scaled.reshape(1, window, 1)
            lstm_pred_scaled = lstm_model.predict(lstm_input, verbose=0)
            
            if hasattr(lstm_scaler, 'data_range_'):
                scale = lstm_scaler.data_range_[0]
                lstm_pred_raw = lstm_pred_scaled[0][0] * scale
                
                w = model_weights.get("LSTM", 1.0)
                predictions.append(float(lstm_pred_raw))
                weights.append(w)
                print(f"  Model LSTM: {lstm_pred_raw:.2f} (Weight: {w:.1f})")
            else:
                 pass
                 
        except Exception as e:
            print(f"LSTM predict error: {e}")

    if not predictions:
        print("Ensemble failed, returning 0")
        return {"mean": 0.0, "std": 0.0, "predictions": []}

    # Weighted Average
    weighted_mean = np.average(predictions, weights=weights)
    
    # Calculate weighted std just for reference, or use simple std
    # Simple std is fine for uncertainty estimation
    std_dev = np.std(predictions)

    return {
        "mean": weighted_mean,
        "std": std_dev,
        "predictions": predictions
    }


def apply_volatility_cap(prediction: float, last_close: float, 
                         vol: float, cap_multiplier: float = 3.0) -> float:
    max_move = last_close * vol * cap_multiplier
    return np.clip(prediction, -max_move, max_move)
