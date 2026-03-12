import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)

from utils.preprocess import load_data, get_features_target, fit_scaler, scale_features, FEATURES

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'water_potability.csv')
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'plots')

def train():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Loading data...")
    df = load_data(DATASET_PATH)
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train_s = scale_features(X_train, scaler)
    X_test_s  = scale_features(X_test,  scaler)

    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    y_pred      = rf.predict(X_test_s)
    y_prob      = rf.predict_proba(X_test_s)[:, 1]
    acc         = accuracy_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=['Not Potable','Potable']))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=cv, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Feature Importance ---
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(FEATURES)), importances[idx], color='steelblue', alpha=0.85)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in idx], rotation=45, ha='right')
    ax.set_title('Feature Importances – Random Forest Classifier')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Not Potable','Potable'])
    ax.set_yticklabels(['Not Potable','Potable'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='teal', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'r--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

    joblib.dump(rf, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Plots saved → {PLOTS_DIR}/")
    return rf, acc, roc_auc

if __name__ == '__main__':
    train()
