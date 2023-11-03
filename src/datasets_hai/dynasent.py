import os
import pickle
import sys

sys.path.append("..")
sys.path.append("../utils")
import random

import numpy as np
import xgboost as xgb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils.utils import loss_01
from xgboost import XGBClassifier


class DynaSentDataset:
    def __init__(
        self, data_x, data_y, ai_preds, ai_scores, hum_preds, ids, sentences, metric_y
    ):
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y, dtype="int")
        self.ai_preds = np.array(ai_preds, dtype="int")
        self.ai_scores = ai_scores
        self.hum_preds = np.array(hum_preds, dtype="int")
        self.ids = ids
        self.sentences = sentences
        self.metric_y = metric_y
        self.get_optimal_deferral()

    def get_optimal_deferral(self):
        """
        gets optimal deferral decisions computed emperically
        Return:
            opt_defer: optimal deferral decisions (binary)
        """
        opt_defer_teaching = []
        for ex in range(len(self.hum_preds)):
            score_hum = self.metric_y([self.data_y[ex]], [self.hum_preds[ex]])
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            if score_hum < score_ai:
                opt_defer_teaching.append(0)
            else:
                opt_defer_teaching.append(1)
        self.opt_defer = np.array(opt_defer_teaching)
        return np.array(opt_defer_teaching)

    def __len__(self):
        return len(self.data_y)


def prepare_dynasent_dataset(path_data):
    # r1_dataset = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")
    r2_dataset = load_dataset("dynabench/dynasent", "dynabench.dynasent.r2.all")

    sentences = []
    data_y = []
    hum_preds = []
    ai_preds = []
    ai_scores = []
    int_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
    for i in range(len(r2_dataset["train"])):
        sentences.append(r2_dataset["train"][i]["sentence"])
        label_distr = r2_dataset["train"][i]["label_distribution"]
        # label_distr is a dictionary with keys 'positive', 'negative', 'neutral', 'mixed'
        # and values being a list of annotators who labeled the sentence with that label
        labels_humans = []
        # iterate through the keys of the dictionary
        for label in label_distr:
            # if the list of annotators is not empty
            if label == "mixed":
                continue
            if label_distr[label]:
                # add the label to the list of labels
                for _ in range(len(label_distr[label])):
                    labels_humans.append(label_to_int[label])
        # sample 3 indices from labels_humans
        idx_labels_humans_sampled = random.sample(list(range(len(labels_humans))), 3)
        # sample 3 labels from labels_humans
        labels_humans_sampled = [labels_humans[i] for i in idx_labels_humans_sampled]
        # get the most common label
        label = max(set(labels_humans_sampled), key=labels_humans_sampled.count)
        # sample 1 point from labels_human excluding labels_humans_sampled
        if len(labels_humans) == 3:
            label_human = 0  # label
        else:
            remaining_indices = [
                i
                for i in range(len(labels_humans))
                if i not in idx_labels_humans_sampled
            ]
            idx_label_human = random.sample(remaining_indices, 1)
            label_human = labels_humans[idx_label_human[0]]

        data_y.append(label)
        hum_preds.append(label_human)

    data_y = np.array(data_y)
    hum_preds = np.array(hum_preds)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, batch_size=16)
    embeddings = np.array(embeddings)

    # split data into 10% train and 90% test, get ids
    data_ids = list(range(0, len(embeddings)))
    data_train_ids, data_test_ids = train_test_split(
        data_ids, test_size=0.5, random_state=42
    )

    accuracy = accuracy_score(data_y, hum_preds)
    print("human Accuracy:", accuracy)
    """
    model_ai = LogisticRegression()#xgb.XGBClassifier()

    model_ai.fit(embeddings[data_train_ids], data_y[data_train_ids])

    # Make predictions on the test set
    ai_preds = model_ai.predict(embeddings[data_test_ids])
    ai_scores = model_ai.predict_proba(embeddings[data_test_ids])
    accuracy = accuracy_score( data_y[data_test_ids], ai_preds)
    """
    sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    results = sentiment_pipeline(sentences)
    label_to_int = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

    preds = []
    scores = []

    for result in results:
        preds.append(label_to_int[result["label"]])
        scores.append(result["score"])
    ai_preds = np.array(preds)
    ai_scores = np.array(scores)

    print("Accuracy:", accuracy_score(data_y, ai_preds))

    dynasent_data = DynaSentDataset(
        embeddings,
        data_y,
        ai_preds,
        ai_scores,
        hum_preds,
        data_test_ids,
        np.array(sentences),
        loss_01,
    )

    # create path dynasent
    if not os.path.exists(path_data + "/dynasent"):
        os.makedirs(path_data + "/dynasent")
    pickle.dump(dynasent_data, open(path_data + "/dynasent/dynasent_dataset.pkl", "wb"))
    return dynasent_data
