import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('.')
from backend.utils.data_preprocessor import DataPreprocessor

transformer_data = DataPreprocessor.load_and_prepare_data(
    r'MASTER_DATASET_ALL_10TRANSFORMERS.csv'
)

feat_cols = ['current_load_kW', 'total_solar_capacity', 'utilization_rate',
             'solar_penetration', 'demand_volatility', 'available_headroom', 'export_ratio']

X = transformer_data[feat_cols].fillna(0)
y = (transformer_data['utilization_rate'] < transformer_data['utilization_rate'].median()).astype(int)

X_scaled = StandardScaler().fit_transform(X)

models = {
    'Random Forest'       : RandomForestClassifier(n_estimators=150, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=3),
    'SVM'                 : SVC(kernel='rbf', random_state=42),
    'Logistic Regression' : LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("=" * 65)
print(f"{'Model':<25} {'CV Mean Accuracy':>18} {'Model Accuracy':>16} {'Verdict':>10}")
print("=" * 65)

results = []
for name, model in models.items():
    cv_scores    = cross_val_score(model, X_scaled, y, cv=5)
    cv_mean      = cv_scores.mean() * 100

    model.fit(X_scaled, y)
    y_pred       = model.predict(X_scaled)
    model_acc    = (y_pred == y).mean() * 100

    if model_acc >= 90 and cv_mean >= 70:
        verdict = 'Excellent'
    elif model_acc >= 80 and cv_mean >= 60:
        verdict = 'Good'
    elif model_acc >= 70:
        verdict = 'Fair'
    else:
        verdict = 'Poor'

    results.append((name, cv_mean, model_acc, verdict))
    print(f"{name:<25} {cv_mean:>17.2f}% {model_acc:>15.2f}% {verdict:>10}")

print("=" * 65)

best = max(results, key=lambda x: (x[2] + x[1]) / 2)
print(f"\n✓ Best Model: {best[0]}  |  CV: {best[1]:.2f}%  |  Accuracy: {best[2]:.2f}%")