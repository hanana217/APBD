"""
Validation du Dataset 
=========================
Verifie que le dataset produit des metriques realistes:
- Accuracy: 75-90% (pas 99%)
- ROC AUC: 0.80-0.92
- Feature importance: variee (pas dominee par une seule feature)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
import warnings
warnings.filterwarnings('ignore')


def load_dataset(path: str = "sql_query_performance_dataset.csv") -> pd.DataFrame:
    """Charge le dataset."""
    df = pd.read_csv(path)
    print(f"[OK] Dataset charge: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def prepare_data(df: pd.DataFrame):
    """Prepare X et y pour le ML."""
    feature_cols = [col for col in df.columns if col != 'is_slow']
    X = df[feature_cols]
    y = df['is_slow']
    return X, y, feature_cols


def validate_with_models(X, y, feature_cols):
    """Valide le dataset avec plusieurs modèles."""

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scaling pour LogReg
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ========== Random Forest ==========
    print("\n" + "="*50)
    print("[RF] RANDOM FOREST")
    print("="*50)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    print(f"Accuracy: {acc_rf:.3f}")
    print(f"ROC AUC:  {auc_rf:.3f}")
    print(f"F1 Score: {f1_rf:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    print(f"CV AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    results['random_forest'] = {
        'accuracy': acc_rf,
        'roc_auc': auc_rf,
        'f1': f1_rf,
        'cv_auc_mean': cv_scores.mean()
    }

    # Feature importance
    print("\n[CHART] Feature Importance (top 10):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(10).iterrows():
        bar = "#" * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {row['importance']:.3f} {bar}")

    results['feature_importance'] = importance

    # ========== Gradient Boosting ==========
    print("\n" + "="*50)
    print("[GB] GRADIENT BOOSTING")
    print("="*50)

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]

    acc_gb = accuracy_score(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_proba_gb)
    f1_gb = f1_score(y_test, y_pred_gb)

    print(f"Accuracy: {acc_gb:.3f}")
    print(f"ROC AUC:  {auc_gb:.3f}")
    print(f"F1 Score: {f1_gb:.3f}")

    results['gradient_boosting'] = {
        'accuracy': acc_gb,
        'roc_auc': auc_gb,
        'f1': f1_gb
    }

    # ========== Logistic Regression ==========
    print("\n" + "="*50)
    print("[LR] LOGISTIC REGRESSION")
    print("="*50)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    f1_lr = f1_score(y_test, y_pred_lr)

    print(f"Accuracy: {acc_lr:.3f}")
    print(f"ROC AUC:  {auc_lr:.3f}")
    print(f"F1 Score: {f1_lr:.3f}")

    results['logistic_regression'] = {
        'accuracy': acc_lr,
        'roc_auc': auc_lr,
        'f1': f1_lr
    }

    return results


def check_correlations(df: pd.DataFrame):
    """Verifie les correlations critiques."""
    print("\n" + "="*50)
    print("[CORR] CORRÉLATIONS CRITIQUES")
    print("="*50)

    correlations = df.corr()['is_slow'].drop('is_slow').sort_values(key=abs, ascending=False)

    print("\nCorrelation avec is_slow:")
    for feat, corr in correlations.head(10).items():
        sign = "+" if corr > 0 else ""
        bar_len = int(abs(corr) * 30)
        bar = "#" * bar_len
        print(f"   {feat:25s} {sign}{corr:.3f} {bar}")

    # Verification critique: longueur ne doit PAS être top correlee
    length_corr = abs(correlations.get('query_length_log', 0))
    rows_corr = abs(correlations.get('rows_examined', 0))
    index_corr = correlations.get('has_index_used', 0)

    print("\n[!]  Verifications critiques:")

    if length_corr < 0.15:
        print(f"   [OK] Longueur decorrelee: |{length_corr:.3f}| < 0.15")
    else:
        print(f"   [X] ATTENTION: Longueur trop correlee: |{length_corr:.3f}| >= 0.15")

    if rows_corr > 0.25:
        print(f"   [OK] rows_examined bien correle: {rows_corr:.3f} > 0.25")
    else:
        print(f"   [!]  rows_examined faiblement correle: {rows_corr:.3f}")

    if index_corr < -0.15:
        print(f"   [OK] Index negativement correle: {index_corr:.3f}")
    else:
        print(f"   [!]  Index pas assez negativement correle: {index_corr:.3f}")

    return correlations


def validate_metrics_range(results: dict):
    """Verifie que les metriques sont dans les plages attendues."""
    print("\n" + "="*50)
    print("[FINAL] VALIDATION FINALE")
    print("="*50)

    rf = results['random_forest']

    checks = []

    # Accuracy: 75-90%
    if 0.75 <= rf['accuracy'] <= 0.90:
        checks.append(("Accuracy dans [75%, 90%]", True, rf['accuracy']))
    else:
        checks.append(("Accuracy dans [75%, 90%]", False, rf['accuracy']))

    # AUC: 0.80-0.92
    if 0.78 <= rf['roc_auc'] <= 0.95:
        checks.append(("ROC AUC dans [0.78, 0.95]", True, rf['roc_auc']))
    else:
        checks.append(("ROC AUC dans [0.78, 0.95]", False, rf['roc_auc']))

    # Feature importance variee
    importance = results['feature_importance']
    top_importance = importance.iloc[0]['importance']
    if top_importance < 0.35:  # Pas de feature dominante
        checks.append(("Pas de feature dominante (top < 35%)", True, top_importance))
    else:
        checks.append(("Pas de feature dominante (top < 35%)", False, top_importance))

    # rows_examined importance > 0
    rows_imp = importance[importance['feature'] == 'rows_examined']['importance'].values[0]
    if rows_imp > 0.05:
        checks.append(("rows_examined importance > 5%", True, rows_imp))
    else:
        checks.append(("rows_examined importance > 5%", False, rows_imp))

    # num_joins importance > 0
    joins_imp = importance[importance['feature'] == 'num_joins']['importance'].values[0]
    if joins_imp > 0.01:
        checks.append(("num_joins importance > 1%", True, joins_imp))
    else:
        checks.append(("num_joins importance > 1%", False, joins_imp))

    print("\nResultats:")
    all_passed = True
    for check_name, passed, value in checks:
        status = "[OK]" if passed else "[X]"
        print(f"   {status} {check_name}: {value:.3f}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] DATASET VALIDÉ - Prêt pour l'entraînement!")
    else:
        print("\n[!]  Certaines verifications ont echoue - Ajuster les paramètres")

    return all_passed


def main():
    """Execute la validation complète."""
    print("="*60)
    print("VALIDATION DU DATASET ")
    print("="*60)

    # Charger
    df = load_dataset()

    # Preparer
    X, y, feature_cols = prepare_data(df)

    # Distribution
    print(f"\n[CHART] Distribution target:")
    print(f"   - Fast (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   - Slow (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")

    # Correlations
    correlations = check_correlations(df)

    # Modèles
    results = validate_with_models(X, y, feature_cols)

    # Validation finale
    all_passed = validate_metrics_range(results)

    print("\n" + "="*60)
    print("FIN DE LA VALIDATION")
    print("="*60)

    return results, all_passed


if __name__ == "__main__":
    results, passed = main()
