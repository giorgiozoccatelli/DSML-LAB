import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, GridSearchCV
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

    #Load datasets
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")
    
    #Separate features and target
    target_col = "target"
    X = train.drop(columns=[target_col])
    y = train[target_col]

    #Identify feature groups
    cont_cols = [c for c in X.columns if c.startswith("cont_")]
    ord_cols  = [c for c in X.columns if c.startswith("ord_")]
    cat_cols  = [c for c in X.columns if c.startswith("cat_")]

    #Preprocessing pipelines 
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

    #Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cont_cols),
            ("ord", ordinal_transformer, ord_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    #Base model
    model = Ridge(random_state=SEED)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    #Grid search for hyperparameter tuning
    param_grid = {
        "model__alpha": [5.0, 10.0, 15.0, 20.0, 25.0] #notice model__aplha follows sklearn Pipeline syntax
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print(f"Best alpha: {grid_search.best_params_['model__alpha']}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

    # Retrain best model on full dataset
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)

    # Predict on test set
    preds = best_model.predict(test)

    # Save submission
    submission = pd.DataFrame({
        "index": np.arange(len(preds)),
        "value": preds
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv generated successfully!")

if __name__ == "__main__":
    main()
