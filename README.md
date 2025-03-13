
---
# Random Forest Model - LTD Ridership Prediction 

## Project Overview

This project aims to predict public transit ridership for LTD (Lane Transit District) using machine learning techniques. The models are trained on historical ridership data and various features derived from LTD's internal data.

## Code Notice
Due to confidentiality, we are unable to present the entire codebase. Therefore, we provide the training and testing scripts for viewing purposes. This code will not run in its current form. 

## Data Processing

The data preparation approach includes:

- Combining passenger count data by date
- Creating time-based features:
    - One-hot encoded day of the week
    - Weekend indicator
    - Year, month, day, week of year
- Aggregating metrics from the previous 6 days. Unable to share these metrics for confidentiality sake.

## Model Development

### Classification Approach

- Initial analysis revealed a multi-modal distribution of ridership data
- Used Gaussian Mixture Model to create 5 distinct ridership classes:
    - Class 0
    - Class 1
    - Class 2
    - Class 3
    - Class 4

Please note, that I am unable to share these classification boundaries because it would violate confidentiality. 

### Random Forest Model

- Initial results showed promising performance:
    - Training accuracy: 100%
    - Test accuracy: ~90%
    - Cross-validation metrics:
        - Mean accuracy: 91.32% (±1.33%)
        - Mean precision: 91.51% (±1.32%)
        - Mean recall: 91.32% (±1.33%)
        - Mean F1 score: 91.35% (±1.32%)
    - Pseudo R² score: 0.9819

### Validation on True Hold-Out Data

- Tested on 100 random days from January-December 2024
- Initial accuracy: 33%
- Average accuracy over multiple random tests: 35-40%

## Key Findings

- Model struggles with predicting extreme classes (0 and 4)
- Performance leveled out to approximately ± 9% of the max people per day in prediction error
- Model is lightweight and only requires LTD's internal data
- Prediction can be made using only 6 days of historical data

## Future Improvements

1. **Model Refinement**
    
    - Enhance prediction accuracy by accounting for all confidence levels
    - Explore ensemble methods with multiple random forests
    - Consider adding a linear layer to assign importance to each forest's prediction
2. **Alternative Models**
    
    - XGBoost
    - SVM
    - Statistical forecasting methods (ARIMA, Exponential Smoothing, Prophet)
    - Deep learning models (RNNs) for longer-term exploration
3. **Data Enhancements**
    
    - Add holiday indicators (Christmas, Thanksgiving, etc.)
    - Consider removing COVID years from training data
    - Incorporate external datasets (e.g., gas prices)
4. **Additional Analysis**
    
    - Create heat maps of ridership
    - Analyze highest boarding stations by month
    - Identify stops with most overflow

## Notes

- Initial results may be overly optimistic due to random data splits versus true future prediction
- The model only requires simple in-house LTD data for training and prediction
- Current implementation can predict total boardings within ± 9% of the max people per day. However, much more work can be done to increase the performance. 

---
# Deep Learning Model - Daily Ridership Prediction

## Overview

This contribution presents a deep learning approach for predicting daily total ridership. This model incorporates external data sources such as Weekly Gas Price, various CPI metrics, Daily Weather Data, and Daily Average AQI. The implementation is based on PyTorch and offers an alternative method to forecast transit usage with promising performance metrics.

## Data Processing

- **Data Sources:** Integrates LTD Ridership Data with external datasets.
- **Feature Extraction:** Converts the `DATE` field into additional temporal features (DayOfWeek, Month, Day).
- **Preprocessing Steps:**
  - Boolean features are converted to binary values.
  - Numerical features are standardized using `StandardScaler`.
  - Categorical features are one-hot encoded via scikit-learn's `ColumnTransformer`.
- **Data Splitting:** The dataset is partitioned into training and testing sets, with target values scaled accordingly.

## Model Architecture

- **Dynamic Construction:** The model is built as a feedforward neural network with configurable hidden layers (e.g., [256, 128, 64]).
- **Layer Components:**
  - Each hidden layer includes a linear transformation followed by ReLU activation.
  - Batch Normalization and Dropout (20% rate) are applied to improve generalization.
  - A final linear layer outputs the daily ridership prediction.

### Training & Evaluation

- **Training Process:**
  - Utilizes Mean Squared Error (MSE) as the loss function.
  - Employs the Adam optimizer with learning rate scheduling and early stopping based on performance improvements.
  - Training loss is logged and visualized over epochs.
- **Evaluation Metrics:**
  - **MSE:** 12,392,269.0000
  - **RMSE:** 3520.2655
  - **R²:** 0.9084
- **Visual Outputs:** Generates plots for training loss (`training_loss.png`) and feature importance (`feature_importance.png`).

### Running the Model

1. **Dependencies:** Ensure the following libraries are installed:
   - PyTorch
   - scikit-learn
   - pandas
   - matplotlib
   - numpy
2. **Setup:**
   - Place the dataset file (`complete_data.tsv`) in the `./data/` directory.
3. **Execution:**
   ```bash
   python predict_ride.py
4. **Outputs:** After running the script, you will find:
   - training_loss.png – Plot of training loss over epochs.
   - feature_importance.png – Bar plot of feature importance.
   - model.pth – Saved model weights.
