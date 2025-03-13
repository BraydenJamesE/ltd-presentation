
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
