
# Major_Exam_MLOps

## Objective
Build a complete MLOps pipeline focused on Linear Regression only, integrating training, testing, quantization, Dockerization, and CI/CD — all managed within a single main branch.

## Steps

### 1. Repository Initialization
- Created the main branch with:
  - `.gitignore`
  - `requirements.txt`
  - `README.md`

### 2. Model Training
- Created `src/train.py` for Linear Regression model training.
- After training, the following results were obtained:

```
Initiating model training...
Training complete. Metrics:
R²: 0.5758
MSE: 0.5559
Mean Absolute Error: 0.5332
Max Absolute Error: 9.8753
Model size in kb: 0.6807
```

- Model saved as: `models/linear_regression_model.joblib`

### 3. Testing
- Created `tests/test_train.py` for model unit tests.
- Ran pytest:

```
pytest tests/test_train.py

tests/test_train.py ..... [100%]
================================================================= 5 passed in 1.03s ==================================================================
```

### 4. Quantization
- Implemented manual quantization in `src/quantize.py`.
- Results after quantization:

```
Initiating parameter quantization...
Loading model parameters...
Weights shape: (8,), Bias: -37.023278
Quantizing weights...
Quantizing bias...
Warning: Bias range is zero, mapping to uint8 midpoint
Quantized parameters saved to models/quantized_model_data.joblib
De-quantizing parameters for validation...
Max weight quantization error: 0.00441086
Bias quantization error: 0.00000000
Loading test data for evaluation...

Evaluation Metrics
R² Score                      -46.6835
Mean Squared Error (MSE)      62.4848
Quantized Model Size          0.5 KB

Inference Test (first 10 samples):
Original predictions (sklearn):
[0.719123 1.764017 2.709659 2.838926 2.604657 2.011754 2.6455   2.168755
 2.740746 3.915615]

Manual original predictions:
[0.719123 1.764017 2.709659 2.838926 2.604657 2.011754 2.6455   2.168755
 2.740746 3.915615]

Manual dequantized predictions:
[-5.445777 -5.15347  -3.241192 -4.624138 -2.219364 -8.362151 -0.520035
 -2.441863 -1.903219 -0.458378]

Difference: Sklearn vs Manual Original:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

Difference: Original Manual vs Dequantized Manual:
[ 6.1649    6.917486  5.950851  7.463064  4.824021 10.373905  3.165535
  4.610618  4.643965  4.373993]
```

### 5. Predictions
- Created `src/predict.py` to test predictions.
- Output after running:

```
Loading trained model...
Loading test dataset...
Making predictions...

Model Performance:
R² Score: 0.5758
Mean Squared Error: 0.5559

Sample Predictions (first 10):
True: 0.48 | Predicted: 0.72 | Diff: 0.24
True: 0.46 | Predicted: 1.76 | Diff: 1.31
True: 5.00 | Predicted: 2.71 | Diff: 2.29
True: 2.19 | Predicted: 2.84 | Diff: 0.65
True: 2.78 | Predicted: 2.60 | Diff: 0.18
True: 1.59 | Predicted: 2.01 | Diff: 0.42
True: 1.98 | Predicted: 2.65 | Diff: 0.66
True: 1.57 | Predicted: 2.17 | Diff: 0.59
True: 3.40 | Predicted: 2.74 | Diff: 0.66
True: 4.47 | Predicted: 3.92 | Diff: 0.55

Prediction completed successfully.
```

### 6. Performance Comparison

| Metric        | Original Model | Quantized Model | Difference |
|---------------|---------------|----------------|------------|
| R² Score      | 0.5758        | -46.6835        | -47.2593   |
| MSE           | 0.5559        | 62.4848         | +61.9289   |
| Model Size    | 0.68 KB       | 0.5 KB          | -0.18 KB   |

### 7. Docker Integration and CI/CD
- Created `Dockerfile` for containerizing the pipeline.
- GitHub Actions workflow (`ci.yml`) configured to:
  - Run tests
  - Train and quantize model
  - Build and test Docker image
  - Push Docker image to Docker Hub

Docker commands used:
```
docker build -t major-exam-mlops .
docker run --rm major-exam-mlops
```
