import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, mean_squared_error, r2_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_cleaner import clean_and_engineer
from src.database import get_latest_trending
from config import MODELS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES = [
    "view_count", "like_count", "comment_count",
    "duration_minutes", "is_short", "is_hd",
    "publish_hour", "publish_month", "is_weekend",
    "like_view_ratio", "comment_view_ratio",
    "engagement_score", "tag_count",
    "title_length", "title_word_count",
    "title_has_number", "title_has_caps",
    "title_exclamation", "title_question",
    "title_sentiment", "description_sentiment"
]


def prepare_data():
    logger.info("Loading data from database...")
    df_raw = get_latest_trending(limit=10000)
    df     = clean_and_engineer(df_raw)

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category_name"].astype(str))
    df["country_encoded"]  = le.fit_transform(df["country"].astype(str))

    all_features = FEATURES + ["category_encoded", "country_encoded"]
    df_model     = df[all_features + ["is_trending_high", "view_count"]].dropna()

    logger.info(f"Data ready — {len(df_model)} rows, {len(all_features)} features")
    return df_model, all_features


def train_classifier(df, features):
    logger.info("\n--- Training Trend Classifier ---")

    X = df[features]
    y = df["is_trending_high"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_prob)
    cv_score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    logger.info(f"Accuracy  : {accuracy:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")
    logger.info(f"ROC-AUC   : {auc:.4f}")
    logger.info(f"CV Score  : {cv_score:.4f}")
    print("\n" + classification_report(y_test, y_pred))

    feature_names = (
        model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else list(X.columns)
    )
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    print("\nTop 10 important features:")
    print(importances.head(10))

    return model, {"accuracy": accuracy, "f1": f1, "auc": auc, "cv": cv_score}


def train_view_predictor(df, features):
    logger.info("\n--- Training View Count Predictor ---")

    X = df[features]
    y = np.log1p(df["view_count"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    logger.info(f"RMSE (log scale) : {rmse:.4f}")
    logger.info(f"R² Score         : {r2:.4f}")

    return model, scaler, {"rmse": rmse, "r2": r2}


def train_clustering(df):
    logger.info("\n--- Training Content Clustering ---")

    tfidf  = TfidfVectorizer(max_features=100, stop_words="english")
    X      = tfidf.fit_transform(df["title"].fillna("")).toarray()

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df     = df.copy()
    df["cluster"] = kmeans.fit_predict(X)

    for i in range(5):
        cluster_titles = df[df["cluster"] == i]["title"].head(3).tolist()
        logger.info(f"Cluster {i}: {cluster_titles}")

    return kmeans, tfidf


def save_models(classifier, view_predictor, scaler, kmeans, tfidf):
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(classifier,     f"{MODELS_PATH}classifier.pkl")
    joblib.dump(view_predictor, f"{MODELS_PATH}view_predictor.pkl")
    joblib.dump(scaler,         f"{MODELS_PATH}scaler.pkl")
    joblib.dump(kmeans,         f"{MODELS_PATH}kmeans.pkl")
    joblib.dump(tfidf,          f"{MODELS_PATH}tfidf.pkl")
    logger.info(f"All models saved to {MODELS_PATH}")


if __name__ == "__main__":
    df, features = prepare_data()

    classifier, clf_metrics             = train_classifier(df, features)
    view_predictor, scaler, reg_metrics = train_view_predictor(df, features)

    logger.info("Loading full data for clustering...")
    df_raw_full   = get_latest_trending(limit=10000)
    df_full       = clean_and_engineer(df_raw_full)
    kmeans, tfidf = train_clustering(df_full)

    save_models(classifier, view_predictor, scaler, kmeans, tfidf)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Classifier Accuracy : {clf_metrics['accuracy']:.2%}")
    print(f"Classifier F1 Score : {clf_metrics['f1']:.2%}")
    print(f"Classifier ROC-AUC  : {clf_metrics['auc']:.2%}")
    print(f"View Predictor R²   : {reg_metrics['r2']:.2%}")
    print("Models saved to models/saved/")
    print("========================================")