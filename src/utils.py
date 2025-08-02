import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_data_split():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    return train_test_split(X, y, test_size=0.2, random_state=42)