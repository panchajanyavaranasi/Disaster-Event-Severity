# src/preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline(num_features, cat_features):
    """
    Creates a reusable ColumnTransformer for numerical and categorical data.
    """
    if len(num_features) == 0:
        raise ValueError("No numeric features provided")

    if len(cat_features) == 0:
        raise ValueError("No categorical features provided")

    # 1. Define transformations for numerical columns
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing values
        ('scaler', StandardScaler())                  # Scale features
    ])

    # 2. Define transformations for categorical columns
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing values
        ('encoder', OneHotEncoder(handle_unknown='ignore'))   # Encode to numeric vectors
    ])

    # 3. Combine them using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )
    
    return preprocessor