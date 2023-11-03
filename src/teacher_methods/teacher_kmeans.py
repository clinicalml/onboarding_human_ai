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

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm import tqdm
from utils.metrics_hai import compute_metrics
from utils.utils import *

from .base_teacher import *


class TeacherKmeans(BaseTeacher):
    """kmeans teacher"""

    def __init__(
        self, data_x, data_y, hum_preds, ai_preds, metric_y, teaching_points=10
    ):
        """Init function.
        Args:
            data_x: 2d numpy array of the features
            data_y: 1d numpy array of labels
            hum_preds:  1d array of the human predictions
            ai_preds:  1d array of the AI predictions
            delta: minimum gain of each region over the prior as raw number of points
            metric_y: metric to use for computing the gain
            teaching_points: number of teaching points to return

        """
        self.data_x = data_x
        self.data_y = data_y
        self.hum_preds = hum_preds
        self.data_y = data_y
        self.ai_preds = ai_preds
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
        self.kmeans = KMeans(n_clusters=self.teaching_points, random_state=0).fit(
            self.data_x
        )
        # get the centers
        self.centers = self.kmeans.cluster_centers_
        # get the labels
        self.kmean_labels = self.kmeans.labels_
        # for cluster in kmeans, compute most optimal deferral decision
        self.teaching_set = []
        for i in range(self.teaching_points):
            # get the indices of the points in the cluster
            cluster_indices = np.where(self.kmean_labels == i)[0]
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
            center = self.centers[i]
            self.teaching_set.append([i, center, best_defer_choice])

    def get_defer_preds(self, data_x, ai_info=None):
        defer_preds = np.zeros(len(data_x))
        # predict the labels for the data
        kmean_labels = self.kmeans.predict(data_x)
        for i in range(len(data_x)):
            # get the cluster label
            cluster_label = kmean_labels[i]
            most_common_defer = self.teaching_set[cluster_label][2]
            defer_preds[i] = most_common_defer
        return defer_preds

    def get_region_labels(self, data_x, ai_info=None):
        # predict the labels for the data
        region_labels = self.kmeans.predict(data_x)
        return np.array(region_labels)

    def get_region_labels_probs(self, data_x, ai_info=None):
        # get scores for each cluster
        scores = self.kmeans.transform(data_x)
        return np.array(scores)
