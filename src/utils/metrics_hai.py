import numpy as np


def get_metrics(preds, truths, metric_y):
    # to be implemented for each method, higher better
    """
    preds: array of predictions
    truths:  target array
    """
    acc = metric_y(truths, preds)  # metrics.accuracy_score(truths, preds)
    metrics_computed = {"score": acc}
    return metrics_computed


def compute_metrics(
    human_preds, ai_preds, reject_decisions, truths, metric_y, to_print=False
):
    coverage = 1 - np.sum(reject_decisions) / len(reject_decisions)
    humanai_preds = []
    human_preds_sys = []
    truths_human = []
    ai_preds_sys = []
    truths_ai = []
    for i in range(len(reject_decisions)):
        if reject_decisions[i] == 1:
            humanai_preds.append(ai_preds[i])
            ai_preds_sys.append(ai_preds[i])
            truths_ai.append(truths[i])
        else:
            humanai_preds.append(human_preds[i])
            human_preds_sys.append(human_preds[i])
            truths_human.append(truths[i])
    humanai_metrics = get_metrics(humanai_preds, truths, metric_y)

    human_metrics = get_metrics(human_preds_sys, truths_human, metric_y)

    ai_metrics = get_metrics(ai_preds_sys, truths_ai, metric_y)

    if to_print:
        print(f"Coverage is {coverage*100:.2f}")
        print(f" metrics of system are: {humanai_metrics}")
        print(f" metrics of human are: {human_metrics}")
        print(f" metrics of AI are: {ai_metrics}")
    return coverage, humanai_metrics, human_metrics, ai_metrics
