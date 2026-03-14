"""
Saves and loads the best trained pipeline.
"""

import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline


DEFAULT_SAVE_DIR  = Path("models")
DEFAULT_FILENAME  = "best_solar_pipeline.pkl"


class ModelSaver:

    def save(self, tuned_models: dict[str, Pipeline],
             best_name: str,
             save_dir: str | Path = DEFAULT_SAVE_DIR,
             filename: str = DEFAULT_FILENAME) -> Path:
        """
        Persist the best model pipeline to disk.

        Args:
            tuned_models : {name: pipeline} from ModelTuner.tune_all()
            best_name    : Key of the best model (top row of evaluator output)
            save_dir     : Directory to save into  (default: models/)
            filename     : File name                (default: best_solar_pipeline.pkl)

        Returns:
            Path to the saved file
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        path = save_dir / filename
        joblib.dump(tuned_models[best_name], path)

        print(f"\n  Saved  : {path}")
        print(f"  Model  : {best_name}")
        print(f"  Target : Efficiency (kWh/kW)")
        print(f"  Contains: StandardScaler + {best_name}")
        return path

    def load(self, path: str | Path = DEFAULT_SAVE_DIR / DEFAULT_FILENAME) -> Pipeline:
        """Load a saved pipeline. Raises FileNotFoundError if missing."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run model_trainer.py first."
            )
        return joblib.load(path)
