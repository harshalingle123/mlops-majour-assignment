import pytest
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error 

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data_split, save_model

@pytest.fixture
def dataset():
    """Fixture to load dataset for tests."""
    return load_data_split()

@pytest.fixture
def trained_model(dataset):
    """Fixture to train a model for tests."""
    X_train, _, y_train, _ = dataset
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def test_data_load(dataset):
    """Test dataset loading and properties."""
    X_train, X_test, y_train, y_test = dataset
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    assert X_train.shape[1] == 8, "Expected 8 features in X_train"
    assert X_test.shape[1] == 8, "Expected 8 features in X_test"
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test), "X_test and y_test length mismatch"
    total = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total
    assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio:.2f} not in [0.75, 0.85]"

def test_model_initialization():
    """Test model creation."""
    model = LinearRegression()
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"
    assert hasattr(model, 'fit'), "Model missing fit method"
    assert hasattr(model, 'predict'), "Model missing predict method"

def test_model_training(trained_model):
    """Test model training attributes."""
    model = trained_model
    assert hasattr(model, 'coef_'), "Model missing coef_ attribute"
    assert hasattr(model, 'intercept_'), "Model missing intercept_ attribute"
    assert model.coef_ is not None, "coef_ is None"
    assert model.intercept_ is not None, "intercept_ is None"
    assert model.coef_.shape == (8,), f"Expected coef_ shape (8,), got {model.coef_.shape}"
    assert isinstance(model.intercept_, (float, np.float64)), "intercept_ not a float"

def test_model_performance(dataset, trained_model):
    """Test model performance metrics."""
    _, X_test, _, y_test = dataset
    model = trained_model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    assert r2 > 0.5, f"R² score {r2:.4f} below threshold of 0.5"
    assert mse > 0, "MSE must be positive"
    print(f"Test Performance: R²={r2:.4f}, MSE={mse:.4f}")

def test_model_persistence(dataset, trained_model, tmp_path):
    """Test model save and load functionality."""
    X_train, X_test, y_train, _ = dataset
    model = trained_model
    model_path = tmp_path / "test_model.joblib"
    save_model(model, str(model_path))
    assert os.path.exists(model_path), "Model file not saved"
    loaded_model = joblib.load(model_path)
    pred_original = model.predict(X_test[:5])
    pred_loaded = loaded_model.predict(X_test[:5])
    np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=6, err_msg="Predictions differ after load")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])