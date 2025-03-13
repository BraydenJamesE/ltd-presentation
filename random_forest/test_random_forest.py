
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import brentq
from scipy.stats import norm, mode
from data.preprocess import get_processed_data
from data.data_analysis import plot_dataFrame, pca, tsne
from data.preprocess import get_guassians


def print_feature_importance(model, data : pd.DataFrame):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order
    print("\nFeature Importance:")
    for idx in sorted_idx:
        print(f"{data.columns[idx]}: {feature_importances[idx]:.4f}")


# TODO: Update function as to check class probs of multiple classes,
#       not just two. 
def find_decision_boundary(class_probs):
    def f(x):
        p0 = w0 * norm.pdf(x, loc=mu0, scale=sigma0)
        p1 = w1 * norm.pdf(x, loc=mu1, scale=sigma1)
        posterior = p1 / (p0 + p1)
        return posterior - target_prob0
    
    sorted_indices = np.argsort(class_probs)
    idx_most_likely_class = np.argmax(class_probs)
    second_most_likely_class = None
    if idx_most_likely_class == 0:
        second_most_likely_class = idx_most_likely_class + 1
    elif idx_most_likely_class == len(class_probs) - 1:
        second_most_likely_class = idx_most_likely_class - 1
    else: 
        if class_probs[idx_most_likely_class - 1] >= class_probs[idx_most_likely_class + 1]:
            second_most_likely_class = idx_most_likely_class - 1
        else: 
            second_most_likely_class = idx_most_likely_class + 1

    
    idx_two_most_likely_classes = sorted_indices[-2:]
    sorted_means, sorted_stds, sorted_weights = get_guassians()

    mu0 = sorted_means[second_most_likely_class]
    mu1 = sorted_means[idx_most_likely_class]
    sigma0 = sorted_stds[second_most_likely_class]
    sigma1 = sorted_stds[idx_most_likely_class]
    w0 = sorted_weights[second_most_likely_class]
    w1 = sorted_weights[idx_most_likely_class]

    target_prob0 = class_probs[idx_most_likely_class]
    
    lower_bound = mu0
    upper_bound = mu1
    if f(lower_bound) * f(upper_bound) > 0:
        return mu1
    x_boundary = brentq(f, lower_bound, upper_bound)
    return x_boundary


def perform_ensemble_testing(data : pd.DataFrame, all_model_paths : list):
    df_last_six_2024 = data.loc[(data['year'] == 2024) & (data['month'] >= 7)]
    df_last_six_2024 = df_last_six_2024.copy()

    df_last_six_2024.loc[:, "date"] = pd.to_datetime(df_last_six_2024[["year", "month", "day"]])
    
    number_of_samples = 100
    size_of_database = df_last_six_2024.shape[0]
    
    if number_of_samples > size_of_database: 
        print(f"\nNotice: Number of samples request ({number_of_samples}) "
              f"is larger than database size ({size_of_database}). "
              f"Setting number of samples to database size.\n"
            )
        number_of_samples = size_of_database

    df_random_n_samples = df_last_six_2024.sample(n=number_of_samples, replace=False)
    
    print(df_random_n_samples.head())

    # Ensuring num of samples is not greater than dataset. If it is, 
    # set the number of samples to the size of the dataset. 
    test_y = df_random_n_samples["ridership_class"]
    total_board_ground_truth = df_random_n_samples["total_board"]
    test_x = df_random_n_samples.drop(columns=["ridership_class", "total_board", "date"])

    ensemble_outputs = {}

    pred_matrices = []
    prob_matrices = []
    for i, model_path in enumerate(all_model_paths):
        print(model_path)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise ValueError(
                f"Error. Invalid model path provided. User provided model path: {model_path}")
        
        class_predictions, class_probs = test_model(df_random_n_samples, 
                                                    model_path, 
                                                    print_results=True,
                                                    create_results_file=True, 
                                                    results_file_name=f"model{i}_results.txt")
        pred_matrices.append(class_predictions.flatten().tolist()) # (num_of_models, number_of_samples)
        prob_matrices.append(class_probs) #  # (num_of_models, number_of_samples, num_of_classes) class_predictions.shape
        
    combined_prob = np.zeros((number_of_samples, 5))

    for i in range(len(all_model_paths)):
        combined_prob += prob_matrices[i]

    row_prob = np.sum(combined_prob, axis=1, keepdims=True)

    ensemble_outputs["final_probs"] = combined_prob / row_prob
    
    row_mode, col_mode = mode(pred_matrices, axis=0)
    print(f"row_mode: {row_mode}")
    ensemble_outputs["final_predictions"] = row_mode

    y_test_pred = ensemble_outputs["final_predictions"]
    probs = ensemble_outputs["final_probs"]
    y_test_pred = np.argmax(probs, axis=1, keepdims=True)
    y_pred_custom = []

    for i, p in enumerate(probs): 
        p4 = p[4] # probability that this sample is class 4
        p0 = p[0] 
        
        if p4 >= 0.2: # If p4 >= 0.2, force class 4, else pick the highest among classes 0..3
            y_pred_custom.append(4.0)
        # elif p0 >= 0.2:
        #     y_pred_custom.append(0.0)
        else:
            y_pred_custom.append(np.argmax(p[:4]))

    test_acc = accuracy_score(test_y, y_test_pred)
    custom_acc = accuracy_score(test_y, y_pred_custom)
    test_acc = accuracy_score(test_y, y_test_pred)

    print("Random Forest Model Performance on Hold Out:")
    print("Accuracy:", test_acc)
    print(classification_report(test_y, y_test_pred, zero_division=0))

    labels = [0,1,2,3,4]
    cm = confusion_matrix(test_y, y_test_pred, labels=labels)
    print("Confusion Matrix:")
    print(cm)


    total_board_est_and_true_total_board_difference_all = []
    total_board_est_and_ground_truth_difference_for_missclassified = []
    total_board_est_and_ground_truth_difference_for_correctly_classified = []
    with open("model_holdout_output.txt", "w") as f:
        for i in range(test_x.shape[0]):
            f.write(" =================== \n")
            sample = test_x.iloc[[i]]  # use double brackets to keep it as a DataFrame
            ground_truth_sample = test_y.iloc[i]
            total_board_ground_truth_value = total_board_ground_truth.iloc[i]
            
            
            # Get the predicted class
            f.write(f"Sample:\n{sample}")
            
            # Get the probability distribution across all classes
            class_probs = ensemble_outputs["final_probs"][i]
            predicted_class = np.argmax(class_probs) # predicted_class = ensemble_outputs["final_predictions"][i]

            # This creates a custom threshold for class 0 and 4. 
            # p4 = class_probs[4]
            # # p0 = class_probs[0]
            # if p4 >= 0.2:
            #     predicted_class = 4.0
            # if p0 >= 0.2:
            #     predicted_class = 0.0

            total_board_est = find_decision_boundary(class_probs)
            total_board_est_and_true_total_board_difference_all.append(abs(total_board_est - total_board_ground_truth_value))
            if predicted_class == ground_truth_sample:
                f.write("✅ Correct Prediction\n")
                total_board_est_and_ground_truth_difference_for_correctly_classified.append(abs(total_board_est - total_board_ground_truth_value))
            else:
                f.write("❌ Incorrect Prediction\n")
                total_board_est_and_ground_truth_difference_for_missclassified.append(abs(total_board_est - total_board_ground_truth_value))
            f.write(f"Total Board Ground Truth: {total_board_ground_truth_value}\n")
            f.write(f"Ground Truth Label: {ground_truth_sample}\n")

            

            f.write(f"Predicted Class: {predicted_class}\n")
            f.write("Confidence Distribution:\n")
            for cls, prob in enumerate(class_probs):
                f.write(f"  Class {cls}: {prob:.2%}\n")
            f.write(f"Total Board Estimate based on classes: {total_board_est}\n")
            f.write(f"Difference in True and Estimated Total Board: ±{abs(total_board_est - total_board_ground_truth_value)}")

            f.write("\n")        
        
        print(f"\nFor Correct Classifications:")
        print(f"Average Difference in Total Board Estimate and True Total Board: ±{np.mean(total_board_est_and_ground_truth_difference_for_correctly_classified)}")
        print(f"Largest Difference in Total Board Estimate and True Total Board: ±{np.max(total_board_est_and_ground_truth_difference_for_correctly_classified)}")
        print(f"Smallest Difference in Total Board Estimate and True Total Board: ±{np.min(total_board_est_and_ground_truth_difference_for_correctly_classified)}")

        print(f"\nFor Incorrect Classifications:")
        print(f"Average Difference in Total Board Estimate and True Total Board: ±{np.mean(total_board_est_and_ground_truth_difference_for_missclassified)}")
        print(f"Largest Difference in Total Board Estimate and True Total Board: ±{np.max(total_board_est_and_ground_truth_difference_for_missclassified)}")
        print(f"Smallest Difference in Total Board Estimate and True Total Board: ±{np.min(total_board_est_and_ground_truth_difference_for_missclassified)}")

        print(f"\nFor All Classifications:")
        print(f"Average Difference in Total Board Estimate and True Total Board: ±{np.mean(total_board_est_and_true_total_board_difference_all)}")
        print(f"Largest Difference in Total Board Estimate and True Total Board: ±{np.max(total_board_est_and_true_total_board_difference_all)}")
        print(f"Smallest Difference in Total Board Estimate and True Total Board: ±{np.min(total_board_est_and_true_total_board_difference_all)}")
        print()
    
    return y_test_pred, probs


def test_model(data : pd.DataFrame, 
               model_path : str, 
               print_results : bool = False, 
               create_results_file : bool = False, 
               results_file_name : str = "model_holdout_output.txt"):
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise ValueError(
            f"Error. Invalid model path provided. User provided model path: {model_path}")
    
    df_random_n_samples = data
    if print_results:
        print(df_random_n_samples.head())

    test_y = df_random_n_samples["ridership_class"]
    total_board_ground_truth = df_random_n_samples["total_board"]
    test_x = df_random_n_samples.drop(columns=["ridership_class", "total_board", "date"])

    y_test_pred = model.predict(test_x)
    probs = model.predict_proba(test_x)  # shape (n_samples, n_classes)
    y_pred_custom = []

    for i, p in enumerate(probs): 
        p4 = p[4] # probability that this sample is class 4
        p0 = p[0] 
        
        if p4 >= 0.2: # If p4 >= 0.2, force class 4, else pick the highest among classes 0..3
            y_pred_custom.append(4.0)
        # elif p0 >= 0.2:
        #     y_pred_custom.append(0.0)
        else:
            y_pred_custom.append(np.argmax(p[:4]))


    test_acc = accuracy_score(test_y, y_test_pred)
    custom_acc = accuracy_score(test_y, y_pred_custom)
    test_acc = accuracy_score(test_y, y_test_pred)
    if print_results:
        # Print metrics
        print("\nCustom Threshold Model Performance")
        print("Custom Accuracy:", custom_acc)
        print("Classification Report (custom threshold):")
        print(classification_report(test_y, y_pred_custom, zero_division=0))
        print("Random Forest Model Performance on Hold Out:")
        print("Accuracy:", test_acc)
        print(classification_report(test_y, y_test_pred, zero_division=0))

        labels = [0,1,2,3,4]
        cm = confusion_matrix(test_y, y_pred_custom, labels=labels)
        print("Confusion Matrix:")
        print(cm)

    if create_results_file:
        total_board_est_and_ground_truth_difference_for_missclassified_entries = []
        with open(results_file_name, "w") as f:
            f.write(f"Model: {model_path}")
            for i in range(test_x.shape[0]):
                f.write(" =================== \n")
                sample = test_x.iloc[[i]]  # use double brackets to keep it as a DataFrame
                ground_truth_sample = test_y.iloc[i]
                total_board_ground_truth_value = total_board_ground_truth.iloc[i]
                predicted_class = model.predict(sample)[0]
                # Get the predicted class
                f.write(f"Sample:\n{sample}")
                
                # Get the probability distribution across all classes
                class_probs = model.predict_proba(sample)[0]

                total_board_est = find_decision_boundary(class_probs)
                total_board_est_and_ground_truth_difference_for_missclassified_entries.append(abs(total_board_est - total_board_ground_truth_value))
                if predicted_class == ground_truth_sample:
                    f.write("✅ Correct Prediction\n")
                else:
                    f.write("❌ Incorrect Prediction\n")
                f.write(f"Total Board Ground Truth: {total_board_ground_truth_value}\n")
                f.write(f"Ground Truth Label: {ground_truth_sample}\n")

                

                f.write(f"Predicted Class: {predicted_class}\n")
                f.write("Confidence Distribution:\n")
                for cls, prob in enumerate(class_probs):
                    f.write(f"  Class {cls}: {prob:.2%}\n")
                f.write(f"Total Board Estimate based on classes: {total_board_est}\n")
                f.write(f"Difference in True and Estimated Total Board: ±{abs(total_board_est - total_board_ground_truth_value)}")

                f.write("\n")        
        
            print(f"Average Difference in Total Board Estimate and True Total Board: ±{np.mean(total_board_est_and_ground_truth_difference_for_missclassified_entries)}")
            print(f"Largest Difference in Total Board Estimate and True Total Board: ±{np.max(total_board_est_and_ground_truth_difference_for_missclassified_entries)}")
            print(f"Smallest Difference in Total Board Estimate and True Total Board: ±{np.min(total_board_est_and_ground_truth_difference_for_missclassified_entries)}")
        
    return y_test_pred, probs


def main():

    # Trained Models. These can be used to create an ensemble estimate.
    model_options = [
                    "saved_models/random_forest/random_forest_model_2.pkl",
                    "saved_models/random_forest/random_forest_model_3.pkl", 
                    "saved_models/random_forest/random_forest_model_favored_class_4.pkl"
                    ]
    
    # Getting the testing dataset 
    df = get_processed_data()
    testing_df = df.loc[(df['year'] == 2024) & (df['month'] >= 1)] 

    perform_ensemble_testing(testing_df, model_options)



if __name__ == "__main__":
    main()