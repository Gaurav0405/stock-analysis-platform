"""
Classification Models Module
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import RANDOM_STATE


class DirectionClassifier:
    """Predict stock direction: UP, DOWN, or FLAT with confidence."""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        )
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
