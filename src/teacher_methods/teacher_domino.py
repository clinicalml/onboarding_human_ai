import sys

import numpy as np

sys.path.append("..")
sys.path.append("../utils")
sys.path.append("./teacher_methods")
import copy
import logging
import math
import multiprocessing
import pickle
import random
import time
from multiprocessing import Pool

import meerkat as mk
from tqdm import tqdm
from utils.metrics_hai import compute_metrics
from utils.utils import *

from .base_teacher import *
from .domino_raw import *
from .domino_raw import DominoSlicer


class TeacherDomino(BaseTeacher):
    """Domino teacher"""

    def __init__(
        self,
        data_x,
        data_y,
        hum_preds,
        ai_preds,
        ai_probs,
        metric_y,
        n_pca_components=10,
        n_mixture_components=10,
        teaching_points=10,
        y_log_likelihood_weight=1,
        y_hat_log_likelihood_weight=1,
    ):
        """Init function.
        Args:
            data_x: 2d numpy array of the features
            data_y: 1d numpy array of labels
            hum_preds:  1d array of the human predictions
            ai_preds:  1d array of the AI predictions
            ai_probs:  2d array of the AI probabilities
            delta: minimum gain of each region over the prior as raw number of points
            metric_y: metric to use for computing the gain
            teaching_points: number of teaching points to return

        """
        self.data_x = data_x
        self.data_y = data_y
        self.hum_preds = hum_preds
        self.data_y = data_y
        self.ai_preds = ai_preds
        self.ai_probs = ai_probs
        self.n_pca_components = n_pca_components
        self.n_mixture_components = n_mixture_components
        self.y_log_likelihood_weight = y_log_likelihood_weight
        self.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.teaching_points = teaching_points
        self.metric_y = metric_y
        logging.info("getting gammas and optimal deferral decisions on teaching set")
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

    def fit(self):
        # run kmeans on the data
        panel = mk.DataPanel(
            {"emb": self.data_x, "target": self.data_y, "pred_probs": self.ai_probs}
        )
        self.domino = DominoSlicer(
            n_slices=self.teaching_points,
            n_pca_components=self.n_pca_components,
            n_mixture_components=self.n_mixture_components,
            y_log_likelihood_weight=self.y_log_likelihood_weight,
            y_hat_log_likelihood_weight=self.y_hat_log_likelihood_weight,
        )
        self.domino.fit(
            data=panel, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        # predict the labels for the data
        domino_preds = self.domino.predict(data=panel, embeddings="emb")
        # covert domino_preds from 1hot to labels
        self.domino_train_preds = np.argmax(domino_preds, axis=1)
        self.teaching_set = []
        for i in range(self.teaching_points):
            # get the indices of the points in the cluster
            cluster_indices = np.where(self.domino_train_preds == i)[0]
            # compute optmal deferral decisions for the cluster
            # get loss if we defer always on cluster
            defer_loss = self.metric_y(
                self.data_y[cluster_indices], self.ai_preds[cluster_indices]
            )
            # get loss if we don't defer always on cluster
            no_defer_loss = self.metric_y(
                self.data_y[cluster_indices], self.hum_preds[cluster_indices]
            )
            if defer_loss < no_defer_loss:
                best_defer_choice = 1
            else:
                best_defer_choice = 0
            # get the center of the cluster
            self.teaching_set.append([i, best_defer_choice])

    def get_defer_preds(self, data_x, ai_info=None):
        defer_preds = np.zeros(len(data_x))
        # predict the labels for the data
        rand_binary = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        rand_binary2 = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        panel = mk.DataPanel(
            {"emb": data_x, "target": rand_binary, "pred_probs": rand_binary2}
        )
        domino_preds = self.domino.predict(data=panel, embeddings="emb")
        # covert domino_preds from 1hot to labels
        domino_preds = np.argmax(domino_preds, axis=1)
        for i in range(len(data_x)):
            defer_pred = self.teaching_set[int(domino_preds[i])][1]
            defer_preds[i] = defer_pred
        return defer_preds

    def get_region_labels(self, data_x, ai_info=None):
        # predict the labels for the data
        rand_binary = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        rand_binary2 = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        panel = mk.DataPanel(
            {"emb": data_x, "target": rand_binary, "pred_probs": rand_binary2}
        )
        domino_preds = self.domino.predict(data=panel, embeddings="emb")
        # covert domino_preds from 1hot to labels
        domino_preds = np.argmax(domino_preds, axis=1)
        return np.array(domino_preds)

    def get_region_labels_probs(self, data_x, ai_info=None):
        # predict the labels for the data
        rand_binary = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        rand_binary2 = self.data_y[
            : len(data_x)
        ]  # np.random.randint(2, size=len(data_x))
        panel = mk.DataPanel(
            {"emb": data_x, "target": rand_binary, "pred_probs": rand_binary2}
        )
        domino_preds = self.domino.predict_proba(data=panel, embeddings="emb")
        return np.array(domino_preds)
