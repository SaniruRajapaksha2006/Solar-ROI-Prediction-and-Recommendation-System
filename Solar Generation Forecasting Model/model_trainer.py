"""
Pipeline:
1. Isolation & Split : Separate by ACCOUNT (households)
2. Physics Baseline  : Compare ML against naive GHI × 0.80 × Days
3. Base Model Compare: Ridge, SVR, RandomForest
4. Hyperparameter Tune: GroupKFold CV respecting household boundaries
5. Overfitting Check : Compare CV-MAE vs Test-MAE for each model
6. Save              : Export best pipeline (predicts Efficiency = kWh/kW)

Key changes from v1:
  - target='Efficiency' (EXPORT_kWh / INV_CAPACITY) — normalises for system size
  - GroupKFold replaces plain KFold in GridSearchCV — no household data leakage
  - max_iter added to Lasso and SVR to suppress convergence warnings
  - Physics baseline (GHI × 0.80 × Days) printed alongside ML metrics
  - Overfitting gap (CV MAE vs Test MAE) reported for every model

Uses GroupShuffleSplit to ensure accounts stay together.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class SolarTrainer:
    
    def __init__(self, df, group_col='ACCOUNT_NO', target='Efficiency'):
        """
        Initialize trainer

        Args:
            df        : Processed DataFrame
            group_col : Column for grouping (default: ACCOUNT_NO)
            target    : Target column (default: 'Efficiency' = EXPORT_kWh / INV_CAPACITY)

        Why Efficiency instead of EXPORT_kWh?
            A 100 kWh error on a 3 kW system is catastrophic (33% error).
            A 100 kWh error on a 10 kW system is acceptable (10% error).
            Predicting Efficiency (kWh) normalises every house to "1 kW",
            forcing the model to learn weather patterns rather than system size.
            At inference: Predicted_kWh = pipeline.predict(X) * INV_CAPACITY
        """
        print("\n" + "="*70)
        print("SOLAR MODEL TRAINER")
        print("="*70)

        self.df = df
        self.group_col = group_col
        self.target = target

        print(f"\nDataset : {len(df):,} records")
        print(f"Accounts: {df[group_col].nunique()}")
        print(f"Target  : {target}  (kWh / kW — multiply by INV_CAPACITY to recover kWh)")

        if target == 'Efficiency' and 'Efficiency' not in df.columns:
            raise ValueError(
                "Column 'Efficiency' not found. "
                "Run data_pipeline.py first - it creates Efficiency = EXPORT_kWh / INV_CAPACITY."
            )
    
    def get_splits(self, test_size=0.2, exclude_cols=None):
        """
        Split data by household (account)

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
    
    def compare_base_models(self, X_train, X_test, y_train, y_test,
                            df_test_raw=None):
        """
        Compare base models (no tuning yet) and benchmark against physics baseline.

        Args:
            X_train, X_test, y_train, y_test: Split data (Efficiency target)
            df_test_raw: Original df rows for test set — used to compute
                         physics baseline MAE. Pass None to skip baseline.

        Returns:
            DataFrame with model comparison
        """
        print("\n" + "-"*70)
        print("STEP 2: BASE MODEL COMPARISON  (target = Efficiency kWh/kW)")
        print("-"*70)

        # ── Physics Baseline ──────────────────────────────────────────────
        if df_test_raw is not None and 'Physics_Pred' in df_test_raw.columns:
            physics_preds = df_test_raw['Physics_Pred']
            physics_mae  = mean_absolute_error(y_test, physics_preds)
            physics_r2   = r2_score(y_test, physics_preds)
            print(f"\nPhysics Baseline (GHI × 0.80 × Days):")
            print(f"   MAE: {physics_mae:.4f} kWh/kW   R²: {physics_r2:.4f}")
            print(f"   ML models MUST beat this to justify complexity.")
            print()
        else:
            physics_mae = None
            print("  (Physics baseline skipped — pass df_test_raw to enable)\n")

        # Define models (conservative params for small dataset)
        models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=10000),       # max_iter: suppress warnings
            "SVR":   SVR(kernel='rbf', C=10, max_iter=5000), # max_iter: suppress warnings
            "RandomForest": RandomForestRegressor(
                n_estimators=50, max_depth=8,
                min_samples_split=5, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=50, max_depth=4,
                learning_rate=0.1, random_state=42
            )
        }

        results = []

        for name, model in models.items():
            print(f"Training {name}...")

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100

            beats_physics = ""
            if physics_mae is not None:
                beats_physics = "✓ beats baseline" if mae < physics_mae else "✗ loses to baseline"

            results.append({
                'Model': name,
                'MAE (kWh/kW)': round(mae, 4),
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2),
                'vs Physics': beats_physics
            })

            print(f"  MAE: {mae:.4f} kWh/kW   R²: {r2:.4f}   {beats_physics}")

        results_df = pd.DataFrame(results).sort_values('MAE (kWh/kW)')

        print("\n" + "-"*70)
        print("BASE MODEL COMPARISON (Efficiency target — lower MAE = better)")
        print("-"*70)
        print(results_df.to_string(index=False))

        return results_df
    
    def hyperparameter_tune_all(self, X_train, y_train, groups_train):
        """
        Tune hyperparameters for ALL models using GroupKFold.

        Overfitting check:
            After fit(), CV MAE (from GridSearchCV) vs Test MAE (from evaluate_tuned_models)
            are printed side-by-side.  A gap > 15 kWh/kW indicates overfitting.

        Args:
            X_train      : Training features
            y_train      : Training target (Efficiency — kWh)
            groups_train : ACCOUNT_NO series aligned with X_train rows

        Returns:
            Dictionary {model_name: best_pipeline}
        """
        print("\n" + "-"*70)
        print("STEP 3: HYPERPARAMETER TUNING — GroupKFold CV")
        print("  (household boundaries respected inside cross-validation)")
        print("-"*70)

        # GroupKFold with 5 folds
        group_kfold = GroupKFold(n_splits=5)

        tuned_models   = {}
        cv_mae_scores  = {}   # store for overfitting check

        # ── Ridge ─────────────────────────────────────────────────────────
        print("\n1. Tuning Ridge...")
        ridge_pipe   = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
        ridge_params = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge_grid   = GridSearchCV(
            ridge_pipe, ridge_params,
            cv=group_kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train, groups=groups_train)
        tuned_models['Ridge']  = ridge_grid.best_estimator_
        cv_mae_scores['Ridge'] = -ridge_grid.best_score_
        print(f"  Best params : {ridge_grid.best_params_}")
        print(f"  CV MAE      : {cv_mae_scores['Ridge']:.4f} kWh/kW")

        # ── Lasso ─────────────────────────────────────────────────────────
        print("\n2. Tuning Lasso...")
        lasso_pipe   = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(max_iter=10000))   # max_iter prevents ConvergenceWarning
        ])
        lasso_params = {'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        lasso_grid   = GridSearchCV(
            lasso_pipe, lasso_params,
            cv=group_kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        lasso_grid.fit(X_train, y_train, groups=groups_train)
        tuned_models['Lasso']  = lasso_grid.best_estimator_
        cv_mae_scores['Lasso'] = -lasso_grid.best_score_
        print(f"  Best params : {lasso_grid.best_params_}")
        print(f"  CV MAE      : {cv_mae_scores['Lasso']:.4f} kWh/kW")

        # ── SVR ───────────────────────────────────────────────────────────
        print("\n3. Tuning SVR...")
        svr_pipe   = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(max_iter=5000))         # max_iter prevents ConvergenceWarning
        ])
        svr_params = {
            'svr__C':       [1, 10, 100],
            'svr__epsilon': [0.01, 0.05, 0.1],
            'svr__kernel':  ['rbf']
        }
        svr_grid = GridSearchCV(
            svr_pipe, svr_params,
            cv=group_kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        svr_grid.fit(X_train, y_train, groups=groups_train)
        tuned_models['SVR']  = svr_grid.best_estimator_
        cv_mae_scores['SVR'] = -svr_grid.best_score_
        print(f"  Best params : {svr_grid.best_params_}")
        print(f"  CV MAE      : {cv_mae_scores['SVR']:.4f} kWh/kW")

        # ── RandomForest ──────────────────────────────────────────────────
        print("\n4. Tuning RandomForest...")
        rf_pipe   = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42))
        ])
        rf_params = {
            'rf__n_estimators':    [50, 100],
            'rf__max_depth':       [6, 8, 10],
            'rf__min_samples_split': [5, 10],
            'rf__min_samples_leaf':  [2, 4]
        }
        rf_grid = GridSearchCV(
            rf_pipe, rf_params,
            cv=group_kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train, groups=groups_train)
        tuned_models['RandomForest']  = rf_grid.best_estimator_
        cv_mae_scores['RandomForest'] = -rf_grid.best_score_
        print(f"  Best params : {rf_grid.best_params_}")
        print(f"  CV MAE      : {cv_mae_scores['RandomForest']:.4f} kWh/kW")

        # ── GradientBoosting ──────────────────────────────────────────────
        print("\n5. Tuning GradientBoosting...")
        gb_pipe   = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42))
        ])
        gb_params = {
            'gb__n_estimators':    [50, 100],
            'gb__max_depth':       [3, 4, 5],
            'gb__learning_rate':   [0.05, 0.1, 0.2],
            'gb__min_samples_split': [5, 10]
        }
        gb_grid = GridSearchCV(
            gb_pipe, gb_params,
            cv=group_kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train, groups=groups_train)
        tuned_models['GradientBoosting']  = gb_grid.best_estimator_
        cv_mae_scores['GradientBoosting'] = -gb_grid.best_score_
        print(f"  Best params : {gb_grid.best_params_}")
        print(f"  CV MAE      : {cv_mae_scores['GradientBoosting']:.4f} kWh/kW")

        print("\n✓ All models tuned with GroupKFold (household-safe CV)")

        self._cv_mae_scores = cv_mae_scores

        return tuned_models
    
    def evaluate_tuned_models(self, tuned_models, X_test, y_test):
        """
        Evaluate all tuned models on test set and report overfitting gap.

        Overfitting check:
            CV MAE   = what GridSearchCV saw during training
            Test MAE = what the model sees on truly unseen households
            Gap      = Test MAE - CV MAE
            If gap > 15 kWh/kW: model is overfitting — reduce complexity.
            If gap < 0         : CV was too optimistic — check data leakage.\

        Args:
            tuned_models : Dictionary of tuned pipelines
            X_test       : Test features
            y_test       : Test target (Efficiency — kWh/kW)

        Returns:
            DataFrame with final comparison
        """
        print("\n" + "-"*70)
        print("STEP 4: FINAL EVALUATION + OVERFITTING CHECK")
        print("  MAE units = kWh/kW  (Efficiency target)")
        print("-"*70)

        cv_scores = getattr(self, '_cv_mae_scores', {})
        results   = []

        for name, pipeline in tuned_models.items():
            preds = pipeline.predict(X_test)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100

            cv_mae = cv_scores.get(name, None)
            gap    = round(mae - cv_mae, 4) if cv_mae is not None else None

            overfit_flag = ""
            if gap is not None:
                if gap > 15:
                    overfit_flag = "OVERFIT"
                elif gap < -5:
                    overfit_flag = "CV leak?"
                else:
                    overfit_flag = "stable"

            results.append({
                'Model':        name,
                'CV MAE':       round(cv_mae, 4) if cv_mae else '-',
                'Test MAE':     round(mae, 4),
                'Gap':          round(gap, 4) if gap is not None else '-',
                'Overfit?':     overfit_flag,
                'RMSE':         round(rmse, 4),
                'R2':           round(r2, 4),
                'MAPE (%)':     round(mape, 2)
            })

        results_df = pd.DataFrame(results).sort_values('Test MAE')

        print("\nFINAL TUNED MODEL COMPARISON:")
        print(results_df.to_string(index=False))

        best = results_df.iloc[0]
        print(f"\nBEST MODEL : {best['Model']}")
        print(f"   Test MAE   : {best['Test MAE']:.4f} kWh/kW")
        print(f"   R²         : {best['R2']:.4f}")
        print(f"   Overfit    : {best['Overfit?']}")
        print(f"\n   Example conversion: {best['Test MAE']:.4f} kWh/kW × 5 kW = "
              f"{best['Test MAE'] * 5:.2f} kWh error for a 5 kW system")

        return results_df
    
    def save_best_model(self, tuned_models, model_name, save_dir='models'):
        """
        Save best model pipeline.

        Args:
            tuned_models : Dictionary of tuned models
            model_name   : Name of best model
            save_dir     : Directory to save
        """
        print("\n" + "-"*70)
        print("STEP 5: SAVE BEST MODEL")
        print("-"*70)

        os.makedirs(save_dir, exist_ok=True)

        best_pipeline = tuned_models[model_name]
        model_path    = os.path.join(save_dir, 'best_solar_pipeline.pkl')
        joblib.dump(best_pipeline, model_path)

        print(f"Saved: {model_path}")
        print(f"  Model   : {model_name}")
        print(f"  Target  : Efficiency (kWh/kW)")
        print(f"  Contains: StandardScaler + Trained Model")
        print(f"\n  Inference example (predict.py):")
        print(f"    pipeline   = joblib.load('{model_path}')")
        print(f"    efficiency = pipeline.predict(X_new)          # kWh/kW")
        print(f"    kwh        = efficiency * X_new['INV_CAPACITY'] # kWh")


if __name__ == "__main__":
    # Load data
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_PATH  = SCRIPT_DIR / "data" / "processed" / "04_features_engineered.csv"
    df = pd.read_csv(DATA_PATH)

    df['Efficiency'] = df['EXPORT_kWh'] / df['INV_CAPACITY']

    print(f"Loaded: {len(df):,} records")

    # Initialize trainer — target is Efficiency (kWh)
    trainer = SolarTrainer(df, group_col='ACCOUNT_NO', target='Efficiency')

    # 1. Split by household
    X_train, X_test, y_train, y_test = trainer.get_splits(
        test_size=0.2,
        exclude_cols=['YEAR', 'EXPORT_kWh']
    )


    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df['ACCOUNT_NO']
    drop_cols = ['Efficiency', 'ACCOUNT_NO'] + [
        c for c in ['YEAR', 'EXPORT_kWh'] if c in df.columns
    ]
    X_all = df.drop(columns=drop_cols)
    train_idx, _ = next(gss.split(X_all, df['Efficiency'], groups=groups))
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    # 2. Compare base models (pass df_test for physics baseline)
    test_idx_mask = df.index.isin(df.index[~df.index.isin(df.index[train_idx])])
    df_test_raw   = df.iloc[[i for i in range(len(df)) if i not in train_idx]]
    base_results  = trainer.compare_base_models(
        X_train, X_test, y_train, y_test, df_test_raw=df_test_raw
    )

    # 3. Tune all models with GroupKFold
    tuned_models = trainer.hyperparameter_tune_all(X_train, y_train, groups_train)

    # 4. Evaluate tuned models + overfitting check
    final_results = trainer.evaluate_tuned_models(tuned_models, X_test, y_test)

    # 5. Save best model
    best_model_name = final_results.iloc[0]['Model']
    trainer.save_best_model(tuned_models, best_model_name)

    print("\n" + "="*70)
    print("TRAINING COMPLETE ✓")
    print("="*70)
