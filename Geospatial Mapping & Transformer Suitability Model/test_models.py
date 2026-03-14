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