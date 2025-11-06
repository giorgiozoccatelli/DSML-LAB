import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


SEED = 42

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # === 1. Load datasets ===
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")

    # === 2. Separate features and target ===
    target_col = "target"
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # === 3. Identify feature groups ===
    cont_cols = [c for c in X.columns if c.startswith("cont_")]
    ord_cols  = [c for c in X.columns if c.startswith("ord_")]
    cat_cols  = [c for c in X.columns if c.startswith("cat_")]

    # === 4. Preprocessing pipelines ===
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # === 5. Combine preprocessors ===
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cont_cols),
            ("ord", ordinal_transformer, ord_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # === 6. Define model ===
    model = Ridge(alpha=1.0, random_state=SEED)

    # === 7. Build pipeline ===
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # === 8. Optional validation ===
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipeline.fit(X_train, y_train)
    y_pred_val = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Validation RMSE: {rmse:.4f}")

    # === 9. Retrain on full training data ===
    pipeline.fit(X, y)

    # === 10. Predict on test set ===
    preds = pipeline.predict(test)

    # === 11. Save submission ===
    submission = pd.DataFrame({
        "index": np.arange(len(preds)),
        "value": preds
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv generated successfully!")

if __name__ == "__main__":
    main()
