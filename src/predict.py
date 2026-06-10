import os
import joblib
import pandas as pd

class ModelInference:

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join('models', 'pipeline.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model pipeline found at {model_path}. Run train.py first."
            )

        self.pipeline = joblib.load(model_path)

    # -----------------------------
    # FEATURE ENGINEERING (CRITICAL FIX)
    # -----------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["loss_per_person"] = df["estimated_economic_loss_usd"] / (df["affected_population"] + 1)

        df["response_ratio"] = df["response_time_hours"] / (df["infrastructure_damage_index"] + 0.1)

        return df

    # -----------------------------
    # PREDICT
    # -----------------------------
    def predict(self, input_data: pd.DataFrame):

        input_data = self._engineer_features(input_data)

        predictions = self.pipeline.predict(input_data)
        return predictions

    # -----------------------------
    # PROBABILITY
    # -----------------------------
    def predict_proba(self, input_data: pd.DataFrame):

        input_data = self._engineer_features(input_data)

        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(input_data)

        return None