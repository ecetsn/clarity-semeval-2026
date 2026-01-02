"""
Classifier definitions and training
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import numpy as np

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_classifier_dict(random_state: int = 42) -> Dict[str, Any]:
    """
    Get dictionary of classifiers (same as siparismaili01)
    
    Returns:
        Dictionary mapping classifier name -> classifier instance
    """
    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
                solver="lbfgs"
            ))
        ]),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                max_iter=1000,
                random_state=random_state,
                dual=False,
                class_weight="balanced"
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
            class_weight="balanced_subsample"
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=random_state,
                verbose=False
            ))
        ]),
    }
    
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss'
        )
    
    if LIGHTGBM_AVAILABLE:
        import os
        os.environ["LIGHTGBM_VERBOSE"] = "-1"
        classifiers["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    
    return classifiers


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    classifiers: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple classifiers and return trained models + predictions
    
    Args:
        X_train: Training features (N, F)
        y_train: Training labels (N,)
        X_dev: Dev features (M, F)
        y_dev: Dev labels (M,)
        classifiers: Dict of classifiers (or None to use default)
        random_state: Random seed
    
    Returns:
        Dictionary mapping classifier_name -> {
            'model': trained_model,
            'train_pred': predictions on train,
            'dev_pred': predictions on dev,
            'train_proba': probabilities on train,
            'dev_proba': probabilities on dev
        }
    """
    if classifiers is None:
        classifiers = get_classifier_dict(random_state=random_state)
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predictions
        train_pred = clf.predict(X_train)
        dev_pred = clf.predict(X_dev)
        
        # Probabilities (if available)
        try:
            train_proba = clf.predict_proba(X_train)
            dev_proba = clf.predict_proba(X_dev)
        except AttributeError:
            # Some classifiers don't have predict_proba
            train_proba = None
            dev_proba = None
        
        results[name] = {
            'model': clf,
            'train_pred': train_pred,
            'dev_pred': dev_pred,
            'train_proba': train_proba,
            'dev_proba': dev_proba
        }
    
    return results

