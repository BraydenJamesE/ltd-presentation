# Daily Ridership Prediction with PyTorch

This repository contains a PyTorch-based model and script designed to predict daily total ridership. The model leverages multiple data sources including LTD Ridership Data, Weekly Gas Price, various CPI Metrics, Daily Weather Data, and Daily Average AQI to make predictions.

---

## Overview

The project consists of:
- **Data Loading & Preprocessing:** Reads a TSV file, processes date features, scales numerical values, encodes categorical variables, and splits the dataset into training and testing sets.
- **Model Architecture:** A feedforward neural network with configurable hidden layers, employing ReLU activation, Batch Normalization, and Dropout regularization.
- **Training & Evaluation:** Utilizes Mean Squared Error (MSE) loss, the Adam optimizer, learning rate scheduling, and early stopping. Evaluation metrics include MSE, RMSE, and R².
- **Feature Importance Analysis:** Estimates feature importance from the first layer’s weights, providing insight into which input features most influence the predictions.

---

## Dataset and Preprocessing

- **Input Data:** Expected in a tab-separated file (`complete_data.tsv`) located in the `./data/` directory.
- **Preprocessing Steps:**
  - Convert the `DATE` column to datetime and extract `DayOfWeek`, `Month`, and `Day`.
  - Split features from the target (`total_board`).
  - Convert boolean columns to binary.
  - Apply standard scaling to numerical features and one-hot encode categorical features using scikit-learn’s `ColumnTransformer`.
  - Split the data into training (80%) and testing (20%) sets.
  - Scale the target variable using `StandardScaler` and convert all data to PyTorch tensors.

---

## Model Architecture

The model is defined in the `RidershipModel` class and is constructed dynamically based on the provided hidden layer dimensions. For example, using hidden layers `[256, 128, 64]`, the architecture includes:
- A sequence of fully-connected layers with ReLU activation.
- Batch Normalization and Dropout (with a probability of 0.2) applied after each hidden layer.
- A final output layer that predicts the daily total ridership.

---

## Training

- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam with an initial learning rate of 0.001.
- **Learning Rate Scheduler:** Reduces the learning rate if no improvement is observed, with early stopping implemented when the learning rate reaches a defined minimum.
- **Logging:** The training loop prints updates every 10 epochs and logs any learning rate adjustments.

The training loss is plotted and saved as `training_loss.png`.

---

## Evaluation

After training, the model is evaluated using:
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² (Coefficient of Determination)**

### Example Results:
- **MSE:** 12,392,269.0000
- **RMSE:** 3520.2655
- **R²:** 0.9084

These metrics indicate a strong predictive performance of the model on the test data.

---

## Feature Importance Analysis

The script includes a feature importance analysis function that:
- Extracts the absolute mean weights from the first linear layer.
- Maps these weights to the corresponding preprocessed feature names.
- Outputs the top features in a sorted order and saves a bar plot (`feature_importance.png`) showing the relative importance of each feature.

---

## How to Run

1. **Prerequisites:** Ensure you have the following dependencies installed:
   - Python 3.6+
   - [PyTorch](https://pytorch.org/)
   - [scikit-learn](https://scikit-learn.org/)
   - [pandas](https://pandas.pydata.org/)
   - [matplotlib](https://matplotlib.org/)
   - [numpy](https://numpy.org/)

2. **Data:** Place your dataset file `complete_data.tsv` in the `./data/` directory.

3. **Execution:**
   ```bash
   python predict_ride.py

4. **Outputs:** After running the script, you will find:
   - training_loss.png – Plot of training loss over epochs.
   - feature_importance.png – Bar plot of feature importance.
   - model.pth – Saved model weights.