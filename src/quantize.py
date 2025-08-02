import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import save_model, load_data_split

def perform_quantization():
    print("Initiating parameter quantization...")
    
    try:
        # Load model parameters
        print("Loading model parameters...")
        params = joblib.load('models/raw_params.joblib')
        weights = params.get('weights')
        bias = params.get('bias')
        
        if weights is None or bias is None:
            raise ValueError("Model parameters 'weights' or 'bias' not found in raw_params.joblib")
        
        print(f"Weights shape: {weights.shape}, Bias: {bias:.6f}")
        
        print("Quantizing weights...")
        w_min, w_max = np.min(weights), np.max(weights)
        w_range = w_max - w_min
        
        if w_range == 0:
            print("Warning: Weight range is zero, mapping to zero uint8 array")
            w_quantized = np.zeros_like(weights, dtype=np.uint8)
        else:
            w_quantized = ((weights - w_min) / w_range * 255).astype(np.uint8)
        
        print("Quantizing bias...")
        b_min, b_max = bias, bias  
        if b_max == b_min:
            print("Warning: Bias range is zero, mapping to uint8 midpoint")
            b_quantized = np.array(128, dtype=np.uint8)
        else:
            b_quantized = ((bias - b_min) / (b_max - b_min) * 255).astype(np.uint8)
        
        quantized_data = {
            'quantized_weights': w_quantized,
            'quantized_bias': b_quantized,
            'weight_min': w_min,
            'weight_max': w_max,
            'bias_min': b_min,
            'bias_max': b_max
        }
        save_path = 'models/quantized_model_data.joblib'
        save_model(quantized_data, save_path)
        print(f"Quantized parameters saved to {save_path}")
        
        print("De-quantizing parameters for validation...")
        if w_range == 0:
            w_dequantized = np.copy(weights)
        else:
            w_dequantized = w_min + (w_quantized * (w_max - w_min) / 255)
        
        b_dequantized = b_min if b_max == b_min else b_min + (b_quantized * (b_max - b_min) / 255)
        
        weight_error = np.abs(weights - w_dequantized).max()
        bias_error = np.abs(bias - b_dequantized)
        print(f"Max weight quantization error: {weight_error:.8f}")
        print(f"Bias quantization error: {bias_error:.8f}")
        
        print("Loading test data for evaluation...")
        _, X_test, _, y_test = load_data_split()
        
        if X_test.shape[1] != w_dequantized.shape[0]:
            raise ValueError(f"Shape mismatch: X_test ({X_test.shape[1]}) vs weights ({w_dequantized.shape[0]})")
        
        y_pred = np.dot(X_test, w_dequantized) + b_dequantized
        mse = mean_squared_error(y_test, y_pred)
        print(f"De-quantized model MSE: {mse:.4f}")
        
        return w_dequantized, b_dequantized
    
    except FileNotFoundError as e:
        print(f"Error: Could not find model file - {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred during quantization: {e}")
        return None, None

if __name__ == "__main__":
    w_dequant, b_dequant = perform_quantization()
    if w_dequant is not None and b_dequant is not None:
        print("Quantization completed successfully.")
    else:
        print("Quantization failed.")