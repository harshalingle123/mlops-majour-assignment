import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import save_model
import os

def get_file_size_kb(model_path):
    return os.path.getsize(model_path) / 1024

def train():
    print("Initiating model training...")
    
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    max_error = np.abs(y_test - y_pred).max()
    model_size_kb = get_file_size_kb("models/hosing_linear_regression_model.joblib")

    
    print(f"Training complete. Metrics:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Max Absolute Error: {max_error:.4f}")
    print(f"Model size in kb: {model_size_kb:.4f}")
    
    # Save model and parameters
    save_model(model, 'models/hosing_linear_regression_model.joblib')
    save_model({'weights': model.coef_, 'bias': model.intercept_}, 'models/raw_params.joblib')
    
    return model, X_test, y_test, r2, mse

if __name__ == "__main__":
    train()