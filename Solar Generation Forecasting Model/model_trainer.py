"""
Pipeline:
1. Isolate: Separate target from features
2. Split: Train/Test partition
3. Scale: Normalize features
4. Fit: Train the model
5. Evaluate: Test performance
6. Save: Export artifacts (.pkl files)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    
    def __init__(self, processed_data_path: str | Path, model_type='RandomForest'):
        print("\n" + "="*70)
        print("MODEL TRAINING PIPELINE")
        print("="*70)

        # Load data
        path = Path(processed_data_path)
        print(f"\nLoading data: {path}")
        self.df = pd.read_csv(path)
        print(f"Loaded: {self.df.shape[0]:,} records, {self.df.shape[1]} columns")
        
        self.scaler = StandardScaler()
        
        self.model_type = model_type
        self.model = self._get_model(model_type)
        print(f"Model: {model_type}")
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.predictions = None
    
    def _get_model(self, model_type):
        """Get model instance based on type"""
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': None,
            'Ridge': None
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def execute_pipeline(self, test_size=0.2):
        # ====================================================================
        # STEP 1: ISOLATION (Separate Target from Features)
        # ====================================================================

        print("\n" + "-" * 70)
        print("STEP 1: ISOLATION")
        print("-" * 70)

        X = self.df.drop(columns=['EXPORT_kWh'])
        y = self.df['EXPORT_kWh']

        print(f"Features (X): {X.shape[1]} columns isolated")
        print(f"Target (y):   EXPORT_kWh isolated")
        print(f"Feature List: {list(X.columns)}")
        
        # ====================================================================
        # STEP 2: SPLIT (Partition into Train/Test)
        # ====================================================================
        
        print("\n" + "-"*70)
        print("STEP 2: SPLIT")
        print("-"*70)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"Training set: {len(self.X_train):,} records ({(1-test_size)*100:.0f}%)")
        print(f"Test set: {len(self.X_test):,} records ({test_size*100:.0f}%)")
        print(f"Split ratio: {1-test_size:.0%} train / {test_size:.0%} test")
        
        # ====================================================================
        # STEP 3: SCALE (Normalize Features - Train Only!)
        # ====================================================================
        
        print("\n" + "-"*70)
        print("STEP 3: SCALE")
        print("-"*70)
        
        print("Fitting scaler on TRAINING data only...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        print("Transforming TEST data using TRAINING statistics...")
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Scaling complete (no data leakage)")
        print(f"  Mean before scaling: {self.X_train.mean().mean():.2f}")
        print(f"  Mean after scaling: {self.X_train_scaled.mean():.2f}")
        
        # ====================================================================
        # STEP 4: FIT (Train the Model)
        # ====================================================================
        
        print("\n" + "-"*70)
        print("STEP 4: FIT")
        print("-"*70)
        
        print(f"Training {self.model_type}...")
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print(f"{self.model_type} trained on {len(self.X_train):,} records")
        
        # ====================================================================
        # STEP 5: PREDICT & EVALUATE
        # ====================================================================
        
        print("\n" + "-"*70)
        print("STEP 5: EVALUATE")
        print("-"*70)
        
        # Predict on test set
        self.predictions = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        r2 = r2_score(self.y_test, self.predictions)
        mape = np.mean(np.abs((self.y_test - self.predictions) / self.y_test)) * 100
        
        print(f"Model Performance on Test Set:")
        print(f"  MAE (Mean Absolute Error): {mae:.2f} kWh")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} kWh")
        print(f"  R² (Coefficient of Determination): {r2:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        
        # Interpretation
        print(f"\nInterpretation:")
        if mae < 50:
            print(f"  Excellent! Predictions within {mae:.0f} kWh on average")
        elif mae < 100:
            print(f"  Good! Predictions within {mae:.0f} kWh on average")
        else:
            print(f" Model could be improved (error = {mae:.0f} kWh)")
        
        if r2 > 0.8:
            print(f"  Excellent fit! Model explains {r2*100:.1f}% of variance")
        elif r2 > 0.6:
            print(f"  Good fit! Model explains {r2*100:.1f}% of variance")
        else:
            print(f" Moderate fit. Model explains {r2*100:.1f}% of variance")
        
        # ====================================================================
        # STEP 6: SAVE ARTIFACTS
        # ====================================================================
        
        print("\n" + "-"*70)
        print("STEP 6: SAVE ARTIFACTS")
        print("-"*70)

        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save scaler
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved: {scaler_path}")

        # Save model
        model_path = models_dir / "solar_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"Saved: {model_path}")

        feature_path = models_dir / "feature_names.pkl"
        joblib.dump(list(X.columns), feature_path)
        print(f"Saved: {feature_path}")
        
        print("\nAll artifacts saved successfully")
        print("  Deploy these 3 files together:")
        print("    • scaler.pkl (normalization)")
        print("    • solar_model.pkl (predictions)")
        print("    • feature_names.pkl (column order)")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70 + "\n")
        
        # Return metrics
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'train_size': len(self.X_train),
            'test_size': len(self.X_test)
        }
    
    def get_feature_importance(self, top_n=10):
        if not hasattr(self.model, 'feature_importances_'):
            print(f"{self.model_type} does not support feature importance")
            return None
        
        feature_names = self.X_train.columns
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-"*70)
        print(importance.head(top_n).to_string(index=False))
        
        return importance


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_PATH = SCRIPT_DIR / "data" / "processed" / "04_features_engineered.csv"

    trainer = ModelTrainer(
        processed_data_path=DATA_PATH,
        model_type='RandomForest'
    )
    
    metrics = trainer.execute_pipeline(test_size=0.2)
    
    # Show feature importance
    importance = trainer.get_feature_importance(top_n=10)
    
    print("\nTraining pipeline complete")
    print(f"  MAE: {metrics['MAE']:.2f} kWh")
    print(f"  R²: {metrics['R2']:.4f}")
