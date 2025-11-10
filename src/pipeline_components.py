from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates engineered datetime features and drops original date columns.
    - days_since_policy = incident_date - policy_bind_date (days)
    - policy_bind_month, incident_month, incident_weekday
    Drops: ['policy_bind_date', 'incident_date'] if present.
    Works directly on a pandas DataFrame.
    """
    def __init__(self,
                 policy_bind_col: str = "policy_bind_date",
                 incident_date_col: str = "incident_date"):
        self.policy_bind_col = policy_bind_col
        self.incident_date_col = incident_date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.policy_bind_col in X.columns and self.incident_date_col in X.columns:
            X[self.policy_bind_col] = pd.to_datetime(X[self.policy_bind_col], errors="coerce")
            X[self.incident_date_col] = pd.to_datetime(X[self.incident_date_col], errors="coerce")
            X["days_since_policy"] = (X[self.incident_date_col] - X[self.policy_bind_col]).dt.days
            X["policy_bind_month"] = X[self.policy_bind_col].dt.month
            X["incident_month"] = X[self.incident_date_col].dt.month
            X["incident_weekday"] = X[self.incident_date_col].dt.weekday
            X = X.drop(columns=[self.policy_bind_col, self.incident_date_col])
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified identifier / leakage columns if present."""
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop = [c for c in self.cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        return X
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates engineered datetime features and drops original date columns.
    - days_since_policy = incident_date - policy_bind_date (days)
    - policy_bind_month, incident_month, incident_weekday
    Drops: ['policy_bind_date', 'incident_date'] if present.
    Works directly on a pandas DataFrame.
    """
    def __init__(self,
                 policy_bind_col: str = "policy_bind_date",
                 incident_date_col: str = "incident_date"):
        self.policy_bind_col = policy_bind_col
        self.incident_date_col = incident_date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.policy_bind_col in X.columns and self.incident_date_col in X.columns:
            X[self.policy_bind_col] = pd.to_datetime(X[self.policy_bind_col], errors="coerce")
            X[self.incident_date_col] = pd.to_datetime(X[self.incident_date_col], errors="coerce")
            X["days_since_policy"] = (X[self.incident_date_col] - X[self.policy_bind_col]).dt.days
            X["policy_bind_month"] = X[self.policy_bind_col].dt.month
            X["incident_month"] = X[self.incident_date_col].dt.month
            X["incident_weekday"] = X[self.incident_date_col].dt.weekday
            X = X.drop(columns=[self.policy_bind_col, self.incident_date_col])
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified identifier / leakage columns if present."""
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop = [c for c in self.cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        return X
