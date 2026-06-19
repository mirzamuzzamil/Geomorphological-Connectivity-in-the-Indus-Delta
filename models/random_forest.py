import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

class RandomForestBaseline:
    """
    Random Forest Baseline model for surface water occurrence classification.
    Relying primarily on static terrain and dynamic meteorological inputs.
    """
    def __init__(self, max_depth=10, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X_train, y_train):
        """
        Fit the Random Forest classifier.
        X_train shape: (num_samples, num_features)
        y_train shape: (num_samples,)
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Predict binary water labels.
        X_test shape: (num_samples, num_features)
        """
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        """
        Predict probability of water occurrence.
        X_test shape: (num_samples, num_features)
        """
        return self.model.predict_proba(X_test)[:, 1]
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate and print classification report.
        """
        preds = self.predict(X_test)
        print("\n--- Random Forest Evaluation ---")
        print(classification_report(y_test, preds))
        return f1_score(y_test, preds, zero_division=0)
