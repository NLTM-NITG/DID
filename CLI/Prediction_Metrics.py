import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve

def main(csv_file_name):
    # Define language mappings for display
    lang2id = {'Marwadi': 0, 'Puneri': 1}
    id2lang = {0: 'Marwadi', 1: 'Puneri'}
    Num_Dialects = 2

    # Read the CSV file
    df = pd.read_csv(csv_file_name)
    predicted_labels = list(df["Predicted"])
    true_labels = list(df["Dialect"])
    predicted_labels = [lang2id[lang] for lang in predicted_labels]
    true_labels = [lang2id[lang] for lang in true_labels]
    print(f"True and predicted labels extracted from: {csv_file_name}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(Num_Dialects))

    # Convert the confusion matrix to a DataFrame for better readability
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[id2lang[i] for i in range(Num_Dialects)], columns=[id2lang[i] for i in range(Num_Dialects)])

    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Recall (Sensitivity or True Positive Rate)
    recall = recall_score(true_labels, predicted_labels, average='macro')

    # Precision
    precision = precision_score(true_labels, predicted_labels, average='macro')

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # False Negative Rate
    fnr = 1 - recall

    # False Positive Rate and True Positive Rate
    fpr = []
    tpr = []
    for i in range(Num_Dialects):
        binarized_true = [1 if label == i else 0 for label in true_labels]
        binarized_pred = [1 if label == i else 0 for label in predicted_labels]
        fpr_class, tpr_class, _ = roc_curve(binarized_true, binarized_pred)
        fpr.append(fpr_class[1])
        tpr.append(tpr_class[1])

    # Calculate average FPR and TPR
    average_fpr = np.mean(fpr)
    average_tpr = np.mean(tpr)

    # Equal Error Rate (EER)
    eer = (average_fpr + fnr) / 2
    print("#######################################################################################")
    # Output the results in percentage
    print(f"Confusion Matrix:\n{conf_matrix_df}")
    print("#######################################################################################")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"True Positive Rate (Recall): {recall * 100:.2f}%")
    print(f"False Positive Rate: {average_fpr * 100:.2f}%")
    print(f"False Negative Rate: {fnr * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print("#######################################################################################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language prediction performance.")
    parser.add_argument('--CSV_File_Name', type=str, help="Path to the CSV file containing predictions", required=True)
    args = parser.parse_args()

    main(args.CSV_File_Name)
