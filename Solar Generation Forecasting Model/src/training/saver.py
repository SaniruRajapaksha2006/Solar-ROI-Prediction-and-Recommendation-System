"""
Saves and loads the best trained pipeline.
"""

import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from typing import Dict, Optional, Union
from utils.utils_config import load_config

DEFAULT_SAVE_DIR = Path("models")


class ModelSaver:

    def save(self, tuned_models: Dict[str, Pipeline],
             best_name: str,
             save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
             filename: Optional[str] = None) -> Path:
        """
        Persist the best model pipeline to disk.

        Args:
            tuned_models : {name: pipeline} from ModelTuner.tune_all()
            best_name    : Key of the best model (top row of evaluator output)
            save_dir     : Directory to save into  (default: models/)
            filename     : File name (default: from config.yaml paths.model)

        Returns:
            Path to the saved file
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = Path(load_config()["paths"]["model"]).name
        path = save_dir / filename
        joblib.dump(tuned_models[best_name], path)

        print(f"\n  Saved  : {path}")
        print(f"  Model  : {best_name}")
        print(f"  Target : Efficiency (kWh/kW)")
        print(f"  Contains: StandardScaler + {best_name}")
        return path

    def load(self, path: Optional[Union[str, Path]] = None) -> Pipeline:
        """Load a saved pipeline. Raises FileNotFoundError if missing."""
        if path is None:
            path = Path(load_config()["paths"]["model"])
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run model_trainer.py first."
            )
        return joblib.load(path)
