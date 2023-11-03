import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def ai_error_analysis(
    dataset, metadata_labels, list_of_metrics=None, min_sample_count=20
):
    """
    dataset is a pandas dataframe with the following columns:
    - 'true_label': the true label of the data point
    - 'predicted_label': the label predicted by the AI
    - 'metadata': array of metadata values for the data point

    metadata_labels is a list of strings that correspond to the metadata
    list_of_metrics: list of metric functions that take in labels and predictions and return a scalar value
    """

    if list_of_metrics is None:
        # default is accuracy, and class wise accuracy
        # count unique labels in the dataset
        unique_labels = np.unique(dataset["true_label"])
        if len(unique_labels) == 2:
            # get confusion matrix
            # from sklearn.metrics import confusion_matrix
            print("binary")

            def confusion_matrix_metric(y_true, y_pred):
                try:
                    tn, fp, fn, tp = confusion_matrix(
                        y_true, y_pred, normalize="true"
                    ).ravel()
                except:
                    tn, fp, fn, tp = [-1, -1, -1, -1]
                return np.array([tn, fp, fn, tp])

            def get_len(truths, preds):
                return len(truths)

            list_of_metrics = [accuracy_score, confusion_matrix_metric, get_len]
        else:
            # get confusion matrix for each class
            def confusion_matrix_metric(y_true, y_pred):
                # only get diagonal elements
                return confusion_matrix(y_true, y_pred, normalize="true").diagonal()

            def get_len(truths, preds):
                return len(truths)

            list_of_metrics = [accuracy_score, get_len]

    # calculate metrics for the whole dataset
    def get_list_of_metrics(df):
        metrics = {}
        for metric in list_of_metrics:
            metric_i = metric(df["true_label"], df["predicted_label"])
            if isinstance(metric_i, np.ndarray):
                for i, label in enumerate(metric_i):
                    metrics[metric.__name__ + "_" + str(i)] = metric_i[i]
            else:
                metrics[metric.__name__] = metric_i
        return metrics

    metrics_overall = get_list_of_metrics(dataset)

    # Define a function to extract an element from the metadata list
    def extract_metadata_element(row, element_index):
        return row["metadata"][element_index]

    # Extract each element of the metadata into a separate column of the DataFrame
    for i, label in enumerate(metadata_labels):
        dataset[label] = dataset.apply(
            lambda row: extract_metadata_element(row, i), axis=1
        )

    # Calculate the error rate, FPR, and sample count for the intersection of each pair of metadata elements

    # Calculate the error rate, FPR, and sample count for each metadata label
    metadata_label_metrics = {}
    #
    metadata_metrics_df = None
    for label in metadata_labels:
        label_metrics = dataset.groupby([label]).apply(
            lambda x: pd.Series(get_list_of_metrics(x))
        )
        metadata_label_metrics[label] = label_metrics
        if metadata_metrics_df is None:
            # add a column to label_metrics that has label in all rows
            # add the name of the row to the dataframe
            label_metrics["subcategory"] = label_metrics.index
            label_metrics["analysis_type"] = "univariate"
            metadata_metrics_df = label_metrics
            metadata_metrics_df["category"] = [label] * len(label_metrics)
        else:
            # add a column to label_metrics that has label in all rows
            label_metrics["subcategory"] = label_metrics.index
            label_metrics["category"] = [label] * len(label_metrics)
            label_metrics["analysis_type"] = "univariate"

            # add the new rows to metadata_metrics_df
            metadata_metrics_df = pd.concat(
                [metadata_metrics_df, label_metrics], ignore_index=True
            )

    # metadata_label_metrics is  a dictionary of pandas dataframes, where the keys are the metadata labels, join them in a signle dataframe

    # do it for each pair of metadata labels
    for i, label1 in enumerate(metadata_labels):
        for label2 in metadata_labels[i + 1 :]:
            label_metrics = dataset.groupby([label1, label2]).apply(
                lambda x: pd.Series(get_list_of_metrics(x))
            )
            label_metrics["subcategory"] = label_metrics.index
            label = label1 + " & " + label2
            label_metrics["category"] = [label] * len(label_metrics)
            label_metrics["analysis_type"] = "pairwise"
            # add the new rows to metadata_metrics_df
            metadata_metrics_df = pd.concat(
                [metadata_metrics_df, label_metrics], ignore_index=True
            )

    # do it for triplets of metadata labels
    for i, label1 in enumerate(metadata_labels):
        for j, label2 in enumerate(metadata_labels[i + 1 :]):
            for label3 in metadata_labels[j + 1 :]:
                label_metrics = dataset.groupby([label1, label2, label3]).apply(
                    lambda x: pd.Series(get_list_of_metrics(x))
                )
                label_metrics["subcategory"] = label_metrics.index
                label_metrics["analysis_type"] = "triplet"
                label = label1 + " & " + label2 + " & " + label3
                label_metrics["category"] = [label] * len(label_metrics)
                # add the new rows to metadata_metrics_df
                metadata_metrics_df = pd.concat(
                    [metadata_metrics_df, label_metrics], ignore_index=True
                )

    # put the category column first and the subcategory column second
    # drop all rows where get_len is less than 20
    metadata_metrics_df = metadata_metrics_df[
        metadata_metrics_df["get_len"] > min_sample_count
    ]
    # add a row for the overall metrics
    overall_df_row = pd.DataFrame(metrics_overall, index=[0])
    overall_df_row["subcategory"] = "overall"
    overall_df_row["category"] = "overall"
    overall_df_row["analysis_type"] = "overall"
    # append overall as first row of dataframe
    metadata_metrics_df = metadata_metrics_df.append(overall_df_row, ignore_index=True)
    # make the last row the first row
    metadata_metrics_df = (
        metadata_metrics_df.iloc[[-1]]
        .append(metadata_metrics_df.iloc[:-1])
        .reset_index(drop=True)
    )

    # for each row, perform a t-test to see if the accuracy_score is significantly different from the overall metric
    # if it is, then add a column that says whether it is significantly different or not
    # if it is not, then add a column that says whether it is significantly different or not
    for i, row in metadata_metrics_df.iterrows():
        if row["analysis_type"] == "overall":
            metadata_metrics_df.loc[i, "significantly_different"] = "overall"
            metadata_metrics_df.loc[i, "p_value"] = 0
        else:
            # do a t-test, and get the p-value, we have accuracy score for the overall dataset, and for the subcategory
            # need to also get the number of samples in the subcategory
            subgroup = [0] * int(
                math.floor(int(row["get_len"]) * (1 - row["accuracy_score"]))
            ) + [1] * int(math.ceil(int(row["get_len"]) * (row["accuracy_score"])))
            overall = [0] * int(
                math.floor(
                    int(metadata_metrics_df.loc[0, "get_len"])
                    * (1 - metadata_metrics_df.loc[0, "accuracy_score"])
                )
            ) + [1] * int(
                math.ceil(
                    int(metadata_metrics_df.loc[0, "get_len"])
                    * (metadata_metrics_df.loc[0, "accuracy_score"])
                )
            )
            t_stat, p_value = stats.ttest_ind(subgroup, overall)
            # add pvale to the dataframe
            metadata_metrics_df.loc[i, "p_value"] = p_value

            if p_value < 0.05:
                metadata_metrics_df.loc[i, "significantly_different"] = "yes"
            else:
                metadata_metrics_df.loc[i, "significantly_different"] = "no"

    cols = metadata_metrics_df.columns.tolist()
    cols = [cols[-4]] + [cols[-3]] + [cols[-5]] + cols[:-5] + [cols[-2]] + [cols[-1]]
    metadata_metrics_df = metadata_metrics_df[cols]
    return metadata_metrics_df
