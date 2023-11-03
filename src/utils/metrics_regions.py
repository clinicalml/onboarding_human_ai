import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from .metrics_hai import *


def get_region_metrics(ground_truth, preds):
    # clustering metrics
    metrics = {}
    metrics["adjusted_rand_score"] = adjusted_rand_score(ground_truth, preds)
    metrics["normalized_mutual_info_score"] = normalized_mutual_info_score(
        ground_truth, preds
    )

    metrics["adjusted_mutual_info_score"] = adjusted_mutual_info_score(
        ground_truth, preds
    )
    metrics["homogeneity_score"] = homogeneity_score(ground_truth, preds)
    metrics["completeness_score"] = completeness_score(ground_truth, preds)
    metrics["v_measure_score"] = v_measure_score(ground_truth, preds)
    metrics["fowlkes_mallows_score"] = fowlkes_mallows_score(ground_truth, preds)
    return metrics


def get_region_stats(region_labels, defer_preds, dataset):
    """
    gets region stats
    Args:
        region_labels: region labels
        defer_preds: defer predictions
        dataset
    Return:
        region_stats: tuple:
            region_stats['region_size']: number of points in each region
            region_stats['region_defer']: deviation from average opt_defer in each region
            basically on average we should defer 80% of the time
            but in this region the optimal deferal is 40%, so the deviation is -0.4

            human-ai team raerall:
            average human-ai team performance
            found a new region, can get performance and deferral,

    """
    random_reject = np.array(
        [np.random.choice([0, 1], p=[0.5, 0.5]) for i in range(len(dataset.data_y))]
    )
    random_human_ai_error = compute_metrics(
        dataset.hum_preds,
        dataset.ai_preds,
        random_reject,
        dataset.data_y,
        dataset.metric_y,
    )[1]["score"]

    region_stats = []
    # get how many unique regions there are
    unique_regions = np.unique(region_labels)

    for region in range(len(unique_regions)):
        region_size = len(region_labels[region_labels == region])
        # get the defer predictions for this region
        region_defer_preds = defer_preds[region_labels == region]
        # get most common defer prediction
        region_defer_pred = np.argmax(np.bincount(region_defer_preds.astype(int)))
        # get AI error in region
        ai_preds = dataset.ai_preds[region_labels == region]
        ai_error = np.mean(ai_preds != dataset.data_y[region_labels == region])
        # get human error in region
        hum_preds = dataset.hum_preds[region_labels == region]
        hum_error = np.mean(hum_preds != dataset.data_y[region_labels == region])

        human_ai_error = compute_metrics(
            hum_preds,
            ai_preds,
            region_defer_preds,
            dataset.data_y[region_labels == region],
            dataset.metric_y,
        )[1]["score"]

        region_deviation = abs(human_ai_error - random_human_ai_error)

        region_info = {
            "size": region_size,
            "deviation": region_deviation,
            "defer_pred": region_defer_pred,
            "ai_error": ai_error,
            "hum_error": hum_error,
        }
        region_stats.append(region_info)
    return region_stats


def plot_regions_size_dev(
    domino_region_stats,
    kmeans_region_stats,
    selection_region_stats,
    gen_region_stats,
    filename=None,
):
    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
    # make the size of the point proportional to the number of points in the region
    sizes = [region["size"] for region in domino_region_stats]
    deviations = [region["deviation"] for region in domino_region_stats]
    plt.scatter(
        sizes, deviations, s=sizes, alpha=0.6, label="Domino", color=CB_color_cycle[0]
    )
    sizes = [region["size"] for region in kmeans_region_stats]
    deviations = [region["deviation"] for region in kmeans_region_stats]
    plt.scatter(
        sizes, deviations, s=sizes, alpha=0.6, label="K-Means", color=CB_color_cycle[1]
    )
    sizes = [region["size"] for region in selection_region_stats[1:]]
    deviations = [region["deviation"] for region in selection_region_stats[1:]]
    plt.scatter(
        sizes,
        deviations,
        s=sizes,
        alpha=0.6,
        label="Selection",
        color=CB_color_cycle[2],
    )
    sizes = [region["size"] for region in gen_region_stats[1:]]
    deviations = [region["deviation"] for region in gen_region_stats[1:]]
    plt.scatter(
        sizes,
        deviations,
        s=sizes,
        alpha=0.6,
        label="Genarative",
        color=CB_color_cycle[3],
    )

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.legend(fontsize="large")
    legend = plt.legend(fontsize="large")
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [100]

    plt.xlabel("Region Size", fontsize="large")
    plt.ylabel("Region Deviation", fontsize="large")
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 7.2
    # increase font sizes
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
