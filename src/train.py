import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessing_pipeline





def train_and_save_model():

    # 1. Load dataset
    data_path = os.path.join('data', 'dataset.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

        # -------------------------
    # FEATURE ENGINEERING (INLINE)
    # -------------------------
    df["loss_per_person"] = df["estimated_economic_loss_usd"] / (df["affected_population"] + 1)

    df["response_ratio"] = df["response_time_hours"] / (df["infrastructure_damage_index"] + 0.1)

    # -------------------------------
    # 2. FINAL TARGET (IMPORTANT FIX)
    # -------------------------------
    target_col = 'severity_level'   # ✅ FIXED

    # -------------------------------
    # 3. FINAL FEATURES (CLEAN)
    # -------------------------------
    FEATURES = [
        "disaster_type",
        "location",
        "latitude",
        "longitude",
        "affected_population",
        "estimated_economic_loss_usd",
        "response_time_hours",
        "aid_provided",
        "infrastructure_damage_index",
        "loss_per_person",
        "response_ratio"
    ]

    X = df[FEATURES]
    y = df[target_col]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 5. Feature separation
    num_features = [
        "latitude",
        "longitude",
        "affected_population",
        "estimated_economic_loss_usd",
        "response_time_hours",
        "infrastructure_damage_index"
    ]

    cat_features = [
        "disaster_type",
        "location",
        "aid_provided"
    ]

    # 6. Preprocessing
    preprocessor = get_preprocessing_pipeline(num_features, cat_features)

    # 7. Model


    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,          # IMPORTANT
        min_samples_split=10, # IMPORTANT
        min_samples_leaf=5,   # IMPORTANT
        class_weight="balanced",
        random_state=42
    )

    # 8. Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # 9. Train
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # 10. Evaluate
    print("Train accuracy:", pipeline.score(X_train, y_train))
    print("Test accuracy:", pipeline.score(X_test, y_test))

    # 11. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/pipeline.pkl")

    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()