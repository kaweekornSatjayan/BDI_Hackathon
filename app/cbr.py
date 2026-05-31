import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from app.config import features_base, features_comorbidity


def apply_weights(X_scaled: np.ndarray, feature_list: list, weight_map: dict) -> np.ndarray:
    """คูณ weight กับ scaled features ก่อนส่งเข้า KNN"""
    weights = np.array([weight_map.get(f, 1.0) for f in feature_list])
    return X_scaled * weights


def load_cleaned_model(file_path: str, specific_features: list, weight_map: dict):
    df = pd.read_excel(file_path)
    all_features = features_base + features_comorbidity + specific_features

    X = df[all_features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = apply_weights(X_scaled, all_features, weight_map)

    k = min(10, len(df))
    knn = NearestNeighbors(n_neighbors=k, metric='manhattan')
    knn.fit(X_weighted)

    return df, X, scaler, knn, all_features


def find_med_column(df: pd.DataFrame, med_name: str) -> str | None:
    exact = f"med_{med_name}"
    if exact in df.columns:
        return exact
    for col in df.columns:
        if med_name.lower() in col.lower():
            return col
    return None


def distance_weighted_rate(med_values: pd.Series, distances: np.ndarray) -> float:
    """
    คำนวณ weighted mean โดยใช้ inverse-distance เป็น weight
    ถ้า distance = 0 (exact match) ให้ weight สูงสุด
    """
    eps = 1e-6  # ป้องกัน division by zero
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum()
    values = med_values.values.astype(float)
    return float(np.dot(weights, values))


def recommendation_label(score: float) -> str:
    if score >= 60:
        return "Recommended"
    if score >= 25:
        return "Consider"
    return "Not Recommended"
