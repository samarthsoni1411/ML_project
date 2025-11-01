# utils/model_trainers.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
import numpy as np
import logging

# Optional: enable clean logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier with safety checks.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_neighbors : int, default=5
        Number of neighbors for KNN

    Returns
    -------
    model : KNeighborsClassifier
        Fitted KNN model
    """
    try:
        # Handle small datasets
        n_neighbors = min(n_neighbors, len(X_train))
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        logging.info(f"KNN trained successfully (neighbors={n_neighbors}, samples={len(X_train)})")
        return model

    except Exception as e:
        logging.error(f"KNN training failed: {e}")
        raise


def train_kmeans(X, n_clusters=2):
    """
    Train a K-Means clustering model with auto-adjusted clusters.

    Parameters
    ----------
    X : array-like
        Input data for clustering
    n_clusters : int, default=2
        Desired number of clusters (auto-adjusted if invalid)

    Returns
    -------
    model : KMeans
        Fitted KMeans model
    """
    try:
        # Adjust cluster count if invalid
        n_clusters = max(1, min(n_clusters, len(X)))

        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
            algorithm="lloyd"  # modern & stable algorithm
        )

        model.fit(X)
        logging.info(f"KMeans trained successfully (clusters={n_clusters}, samples={len(X)})")
        return model

    except ValueError as ve:
        logging.warning(f"⚠️ KMeans parameter issue — {ve}. Retrying with 2 clusters.")
        model = KMeans(n_clusters=2, n_init=10, random_state=42)
        model.fit(X)
        return model

    except Exception as e:
        logging.error(f"KMeans training failed: {e}")
        raise


def safe_predict(model, X):
    """
    Attempt to predict using a trained model.
    Returns None if model is not fitted.
    """
    try:
        return model.predict(X)
    except NotFittedError:
        logging.error("Attempted to predict using an unfitted model.")
        return None
