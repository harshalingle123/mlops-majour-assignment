import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_data_split

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def predict():
    print("Loading trained model...")
    try:
        model = joblib.load('models/hosing_linear_regression_model.joblib')
    except FileNotFoundError as e:
        print(f"Error: Could not find model file - {e}")
        return False

    print("Loading test dataset...")
    try:
        _, X_test, _, y_test = load_data_split()
    except Exception as e:
        print(f"Error: Could not load test dataset - {e}")
        return False

    print("Making predictions...")
    y_pred = model.predict(X_test)

    r2, mse = calculate_metrics(y_test, y_pred)

    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    print("\nSample Predictions (first 10):")
    for i in range(min(10, len(y_test))):
        print(f"True: {y_test[i]:.2f} | Predicted: {y_pred[i]:.2f} | Diff: {abs(y_test[i] - y_pred[i]):.2f}")

    print("\nPrediction completed successfully.")
    return True

if __name__ == "__main__":
    success = predict()
    if success:
        print("Prediction process completed successfully.")
    else:
        print("Prediction process failed.")