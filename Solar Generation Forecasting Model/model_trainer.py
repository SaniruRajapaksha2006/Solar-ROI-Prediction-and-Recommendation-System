"""
src/model_trainer.py

Solar Forecasting Model Trainer with Household-Based Splitting

Pipeline:
1. Isolation & Split: Separate by ACCOUNT (households)
2. Base Model Comparison: Ridge, SVR, RandomForest
3. Hyperparameter Tuning: Optimize ALL models
4. Save: Export best pipeline artifact

Uses GroupShuffleSplit to ensure accounts stay together.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class SolarTrainer:
    """
    Solar forecasting trainer with account-based splitting
    
    Key Design:
    - GroupShuffleSplit: Keeps all months of same account together
    - Pipeline: Ensures scaling only uses train data
    - Hyperparameter tuning: Optimizes for small dataset
    """
    
    def __init__(self, df, group_col='ACCOUNT_NO', target='EXPORT_kWh'):
        """
        Initialize trainer
        
        Args:
            df: Processed DataFrame
            group_col: Column for grouping (default: ACCOUNT_NO)
            target: Target column (default: EXPORT_kWh)
        """
        print("\n" + "="*70)
        print("SOLAR MODEL TRAINER")
        print("="*70)
        
        self.df = df
        self.group_col = group_col
        self.target = target
        
        print(f"\nDataset: {len(df):,} records")
        print(f"Accounts: {df[group_col].nunique()}")
        print(f"Target: {target}")
    
    def get_splits(self, test_size=0.2, exclude_cols=None):
        """
        Split data by household (account)
        
        Ensures all months of Account #123 stay in same set
        
        Args:
            test_size: Fraction for test set (default 0.2)
            exclude_cols: Columns to exclude from features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n" + "-"*70)
        print("STEP 1: ISOLATION & HOUSEHOLD SPLIT")
        print("-"*70)
        
        # Default exclusions
        if exclude_cols is None:
            exclude_cols = ['YEAR']
        
        # 1. ISOLATION
        drop_cols = [self.target, self.group_col] + [
            col for col in exclude_cols if col in self.df.columns
        ]
        
        X = self.df.drop(columns=drop_cols)
        y = self.df[self.target]
        groups = self.df[self.group_col]
        
        print(f"Features: {X.shape[1]} columns")
        print(f"Target: {self.target}")
        print(f"Groups: {self.group_col} ({groups.nunique()} unique accounts)")
        
        # 2. HOUSEHOLD SPLIT (GroupShuffleSplit)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Verify no account overlap
        train_accounts = set(groups.iloc[train_idx].unique())
        test_accounts = set(groups.iloc[test_idx].unique())
        overlap = train_accounts.intersection(test_accounts)
        
        if overlap:
            raise ValueError(f"Data leakage! {len(overlap)} accounts in both sets")
        
        print(f"\nTrain: {len(X_train):,} records, {len(train_accounts)} accounts")
        print(f"Test: {len(X_test):,} records, {len(test_accounts)} accounts")
        print(f"✓ No account overlap (household split verified)")
        
        return X_train, X_test, y_train, y_test
    
    def compare_base_models(self, X_train, X_test, y_train, y_test):
        """
        Compare base models (no tuning yet)
        
        Pipeline ensures scaling only uses train statistics
        
        Args:
            X_train, X_test, y_train, y_test: Split data
            
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "-"*70)
        print("STEP 2: BASE MODEL COMPARISON")
        print("-"*70)
        
        # Define models (conservative params for small dataset)
        models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "SVR": SVR(kernel='rbf', C=10),
            "RandomForest": RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            # Fit
            pipe.fit(X_train, y_train)
            
            # Predict
            preds = pipe.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            })
            
            print(f"  MAE: {mae:.2f} kWh")
            print(f"  R²: {r2:.4f}")
        
        results_df = pd.DataFrame(results).sort_values('MAE')
        
        print("\n" + "-"*70)
        print("MODEL COMPARISON RESULTS")
        print("-"*70)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def hyperparameter_tune_all(self, X_train, y_train):
        """
        Tune hyperparameters for ALL models
        
        Conservative tuning for small dataset (~3000 records)
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with best pipeline for each model
        """
        print("\n" + "-"*70)
        print("STEP 3: HYPERPARAMETER TUNING (ALL MODELS)")
        print("-"*70)
        
        tuned_models = {}
        
        # ====================================================================
        # Ridge Regression
        # ====================================================================
        print("\n1. Tuning Ridge...")
        
        ridge_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        ridge_params = {
            'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
        ridge_grid = GridSearchCV(
            ridge_pipe, ridge_params,
            cv=3,  # 3-fold (conservative for small data)
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        tuned_models['Ridge'] = ridge_grid.best_estimator_
        print(f"  Best params: {ridge_grid.best_params_}")
        print(f"  Best MAE: {-ridge_grid.best_score_:.2f}")
        
        # ====================================================================
        # Lasso Regression
        # ====================================================================
        print("\n2. Tuning Lasso...")
        
        lasso_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso())
        ])
        
        lasso_params = {
            'lasso__alpha': [0.01, 0.1, 1.0, 10.0]
        }
        
        lasso_grid = GridSearchCV(
            lasso_pipe, lasso_params,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        lasso_grid.fit(X_train, y_train)
        tuned_models['Lasso'] = lasso_grid.best_estimator_
        print(f"  Best params: {lasso_grid.best_params_}")
        print(f"  Best MAE: {-lasso_grid.best_score_:.2f}")
        
        # ====================================================================
        # SVR
        # ====================================================================
        print("\n3. Tuning SVR...")
        
        svr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        svr_params = {
            'svr__C': [1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 0.5],
            'svr__kernel': ['rbf']
        }
        
        svr_grid = GridSearchCV(
            svr_pipe, svr_params,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        svr_grid.fit(X_train, y_train)
        tuned_models['SVR'] = svr_grid.best_estimator_
        print(f"  Best params: {svr_grid.best_params_}")
        print(f"  Best MAE: {-svr_grid.best_score_:.2f}")
        
        # ====================================================================
        # Random Forest (Conservative for small dataset)
        # ====================================================================
        print("\n4. Tuning RandomForest...")
        
        rf_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42))
        ])
        
        rf_params = {
            'rf__n_estimators': [50, 100],        # Limited range
            'rf__max_depth': [6, 8, 10],          # Not too deep
            'rf__min_samples_split': [5, 10],     # Prevent overfitting
            'rf__min_samples_leaf': [2, 4]        # Conservative
        }
        
        rf_grid = GridSearchCV(
            rf_pipe, rf_params,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        tuned_models['RandomForest'] = rf_grid.best_estimator_
        print(f"  Best params: {rf_grid.best_params_}")
        print(f"  Best MAE: {-rf_grid.best_score_:.2f}")
        
        # ====================================================================
        # Gradient Boosting (Conservative)
        # ====================================================================
        print("\n5. Tuning GradientBoosting...")
        
        gb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42))
        ])
        
        gb_params = {
            'gb__n_estimators': [50, 100],
            'gb__max_depth': [3, 4, 5],
            'gb__learning_rate': [0.05, 0.1, 0.2],
            'gb__min_samples_split': [5, 10]
        }
        
        gb_grid = GridSearchCV(
            gb_pipe, gb_params,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        tuned_models['GradientBoosting'] = gb_grid.best_estimator_
        print(f"  Best params: {gb_grid.best_params_}")
        print(f"  Best MAE: {-gb_grid.best_score_:.2f}")
        
        print("\n✓ All models tuned")
        
        return tuned_models
    
    def evaluate_tuned_models(self, tuned_models, X_test, y_test):
        """
        Evaluate all tuned models on test set
        
        Args:
            tuned_models: Dictionary of tuned pipelines
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with final comparison
        """
        print("\n" + "-"*70)
        print("STEP 4: FINAL EVALUATION (TUNED MODELS)")
        print("-"*70)
        
        results = []
        
        for name, pipeline in tuned_models.items():
            preds = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            })
        
        results_df = pd.DataFrame(results).sort_values('MAE')
        
        print("\nFINAL TUNED MODEL COMPARISON:")
        print(results_df.to_string(index=False))
        
        # Highlight best
        best = results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['Model']}")
        print(f"   MAE: {best['MAE']:.2f} kWh")
        print(f"   R²: {best['R2']:.4f}")
        
        return results_df
    
    def save_best_model(self, tuned_models, model_name, save_dir='models'):
        """
        Save best model pipeline
        
        Args:
            tuned_models: Dictionary of tuned models
            model_name: Name of best model
            save_dir: Directory to save
        """
        print("\n" + "-"*70)
        print("STEP 5: SAVE BEST MODEL")
        print("-"*70)
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_pipeline = tuned_models[model_name]
        
        # Save complete pipeline (scaler + model together)
        model_path = os.path.join(save_dir, 'best_solar_pipeline.pkl')
        joblib.dump(best_pipeline, model_path)
        
        print(f"✓ Saved: {model_path}")
        print(f"  Model: {model_name}")
        print(f"  Contains: StandardScaler + Trained Model")
        print(f"\n  For inference:")
        print(f"    pipeline = joblib.load('{model_path}')")
        print(f"    predictions = pipeline.predict(new_data)")


if __name__ == "__main__":
    # Load data
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_PATH = SCRIPT_DIR / "data" / "processed" / "04_features_engineered.csv"
    df = pd.read_csv(DATA_PATH)
    
    print(f"Loaded: {len(df):,} records")
    
    # Initialize trainer
    trainer = SolarTrainer(df, group_col='ACCOUNT_NO', target='EXPORT_kWh')
    
    # 1. Split by household
    X_train, X_test, y_train, y_test = trainer.get_splits(
        test_size=0.2,
        exclude_cols=['YEAR']
    )
    
    # 2. Compare base models
    base_results = trainer.compare_base_models(X_train, X_test, y_train, y_test)
    
    # 3. Tune all models
    tuned_models = trainer.hyperparameter_tune_all(X_train, y_train)
    
    # 4. Evaluate tuned models
    final_results = trainer.evaluate_tuned_models(tuned_models, X_test, y_test)
    
    # 5. Save best model
    best_model_name = final_results.iloc[0]['Model']
    trainer.save_best_model(tuned_models, best_model_name)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE ✓")
    print("="*70)
