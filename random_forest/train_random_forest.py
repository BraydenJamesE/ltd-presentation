import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, r2_score
from collections import Counter
from sklearn.model_selection import GridSearchCV
from data.preprocess import get_processed_data
import numpy as np
import joblib



def train_random_forest(X: pd.DataFrame, y: pd.DataFrame, perform_param_cross_validation : bool = False, find_total_board_estimate : bool = False):
    """
    Trains a Random Forest model and returns it.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels

    Returns:
    --------
    model : RandomForestClassifier
        Trained Random Forest model
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    total_board_train = X_train["total_board"]
    X_train.drop(columns=["total_board"], inplace=True)
    total_board_test = X_test["total_board"]
    X_test.drop(columns=["total_board"], inplace=True)

    # Initialize and train the model
    if perform_param_cross_validation:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 10, 15, 25],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', None],
            'class_weight': [
                            None, 
                             'balanced', 
                             'balanced_subsample', 
                            {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                             {0: 1, 1: 1, 2: 1, 3: 1, 4: 3}, 
                             {0: 1, 1: 1, 2: 1, 3: 1, 4: 2}, 
                             {0: 1, 1: 1, 2: 1, 3: 1, 4: 4}, 
                             {0: 2, 1: 1, 2: 1, 3: 1, 4: 2}, 
                             {0: 2, 1: 1, 2: 1, 3: 1, 4: 3},
                             {0: 1, 1: 1, 2: 1, 3: 1, 4: 10} 
                            ]
        }
        initial_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=initial_model,
            param_grid=param_grid,
            scoring='balanced_accuracy',  # TODO: Traing a model on f1, precision, and recall too. 
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Params: {grid_search.best_params_}")
    
    else: 
        model = RandomForestClassifier(
            class_weight="balanced",
            max_depth=10,
            max_features=None,
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=200,
            random_state=42
        )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)  # shape (n_samples, n_classes)

    print("Training class distribution:", Counter(y_train))
    print("Testing class distribution:", Counter(y_test))
          
    y_pred_custom = []
    for p in probs:
        p4 = p[4]  # probability that this sample is class 4
        # If p4 >= 0.2, force class 4, else pick the highest among classes 0..3
        if p4 >= 0.2:
            y_pred_custom.append(4.0)
        else:
            y_pred_custom.append(np.argmax(p[:4]))


    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    custom_acc = accuracy_score(y_test, y_pred_custom)
    print("\nCustom Threshold Model Performance")
    print("Custom Accuracy:", custom_acc)
    print("Classification Report (custom threshold):")
    print(classification_report(y_test, y_pred_custom))

    print("Random Forest Model Performance:")
    print("Training Accuracy:", train_acc)
    print("Accuracy:", test_acc)
    print(classification_report(y_test, y_test_pred))

    # == Performing Cross Validation == 
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    misclassifications = Counter()

    # Store misclassified instances along with total_board values
    misclassified_cases = []
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1_scores = []


    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        X_test_cv_altered = X_test_cv.drop(columns=["total_board"])

        y_pred_cv = model.predict(X_test_cv_altered)

        accuracy = accuracy_score(y_test_cv, y_pred_cv)
        precision = precision_score(y_test_cv, y_pred_cv, average="weighted", zero_division=0)
        recall = recall_score(y_test_cv, y_pred_cv, average="weighted", zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average="weighted", zero_division=0)

        # Store metrics
        cv_accuracies.append(accuracy)
        cv_precisions.append(precision)
        cv_recalls.append(recall)
        cv_f1_scores.append(f1)

        # Track misclassified cases
        for idx, (true_label, pred_label) in enumerate(zip(y_test_cv, y_pred_cv)):
            if true_label != pred_label:
                misclassifications[(true_label, pred_label)] += 1
                misclassified_cases.append((true_label, pred_label))  # Use correctly aligned data
        
                                                                                
    # Print summary of CV performance
    print("\n ⚔️ Cross-Validation Metrics Summary:")
    print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} (± {np.std(cv_accuracies):.4f})")
    print(f"Mean Precision: {np.mean(cv_precisions):.4f} (± {np.std(cv_precisions):.4f})")
    print(f"Mean Recall: {np.mean(cv_recalls):.4f} (± {np.std(cv_recalls):.4f})")
    print(f"Mean F1 Score: {np.mean(cv_f1_scores):.4f} (± {np.std(cv_f1_scores):.4f})")
    print(f"\n⬆️ Highest CV Accuracy: {max(cv_accuracies):.4f}")
    print(f"🔻Lowest CV Accuracy: {min(cv_accuracies):.4f}")

    y_pred_proba = model.predict_proba(X_test)

    # Compute weighted probabilities as a "regression-like" output
    y_pred_regression = (y_pred_proba * np.arange(len(y_pred_proba[0]))).sum(axis=1)

    # Compute pseudo R^2
    r2 = r2_score(y_test, y_pred_regression)
    print(f"Pseudo R^2 Score: {r2:.4f}")

    return model  # Returning the trained model


def main():
    df = get_processed_data() # Getting a dataset from the database

    # Removing the last year out of the dataset for testing later.
    rows_to_drop = df.loc[(df["year"] == 2024) & (df["month"] >= 1)].index 
    df.drop(index=rows_to_drop, inplace=True)

    # Getting features and targets. Note that "total_board" is left in the dataset. This is removed in the train_random_forest model. 
    y = df["ridership_class"]
    X = df.drop(columns=["ridership_class"])
    print(X)

    model = train_random_forest(X, y, perform_param_cross_validation=True) # Getting the model

    # Save the trained model to a file
    model_path = "saved_models/random_forest/temp.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()