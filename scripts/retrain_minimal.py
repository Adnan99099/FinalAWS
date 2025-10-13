"""Minimal retrain script
Fits the project's preprocessing pipeline on the available dataset and trains a
lightweight LinearRegression model. Saves `artifacts/preprocessor.pkl` and
`artifacts/model.pkl` compatible with the current environment.
"""
import os
import sys
import traceback

import pandas as pd
import numpy as np

from src.components.data_transformation import DataTransformation
from sklearn.linear_model import LinearRegression
from src.utils import save_object


def main():
    try:
        # locate dataset - prefer artifacts/raw if present else notebook/data/stud.csv
        possible = [
            os.path.join('artifacts', 'data.csv'),
            os.path.join('notebook', 'data', 'stud.csv')
        ]
        csv_path = None
        for p in possible:
            if os.path.exists(p):
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError('Could not find dataset at artifacts/data.csv or notebook/data/stud.csv')

        df = pd.read_csv(csv_path)
        print('Loaded dataset:', csv_path, 'shape=', df.shape)

        # Split features/target
        target = 'math_score'
        if target not in df.columns:
            raise ValueError(f"Expected target column '{target}' in dataset")

        X = df.drop(columns=[target], axis=1)
        y = df[target]

        dt = DataTransformation()
        preprocessor = dt.get_data_transformer_object()

        print('Fitting preprocessor...')
        X_trans = preprocessor.fit_transform(X)
        print('Preprocessor fitted. transformed shape=', getattr(X_trans, 'shape', None))

        # Train a simple Linear Regression model (fast and dependency-light)
        print('Training LinearRegression...')
        model = LinearRegression()
        model.fit(X_trans, y)
        print('Model trained.')

        # Save artifacts
        os.makedirs('artifacts', exist_ok=True)
        save_object(os.path.join('artifacts', 'preprocessor.pkl'), preprocessor)
        save_object(os.path.join('artifacts', 'model.pkl'), model)
        print('Saved artifacts to artifacts/preprocessor.pkl and artifacts/model.pkl')

    except Exception as e:
        print('Retrain failed:')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
