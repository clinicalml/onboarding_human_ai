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

from base_teacher import *
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm import tqdm
from utils.metrics_hai import compute_metrics
from utils.utils import *

# UPDATE FIRST


class DataRegionModelError:
    def __init__(
        self, indices, text_description, defer_label, label_distr, ai_pred_distr
    ):
        """Init function.
        Args:
            indices: indices of the data points in the region
            text_description: text description of the region
            defer_label: label of the region
            label_distr: distribution of labels in the region
            hum_pred_distr: distribution of human predictions in the region
            ai_pred_distr: distribution of AI predictions in the region
        """

        self.indices = indices
        self.text_description = text_description
        self.defer_label = defer_label
        self.label_distr = label_distr
        self.ai_pred_distr = ai_pred_distr


class TeacherSelectiveModelError:
    """Returns top examples that best teach a learner when to defer to a classifier.
    Given a tabular dataset with classifier predictions, human predictions and a similarity metric,
    the method returns the top k images that best describe when to defer to the AI.


    """

    def __init__(
        self,
        data_x,
        data_y,
        ai_preds,
        prior_rejector_preds,
        sim_kernel,
        metric_y,
        teaching_points=10,
        alpha=0,
        beta_high=0.5,
        beta_low=0.01,
        randomized_sampling=1,
        delta=0,
        parallel_processes=1,
        threshold_metric=0.5,
    ):
        """Init function.
        Args:
            data_x: 2d numpy array of the features
            data_y: 1d numpy array of labels
            ai_preds:  1d array of the AI predictions
            prior_rejector_preds: 1d binary array of the prior rejector preds
            sim_kernel: function that takes as input two inputs and returns a positive number
            metric_y: metric function (positive,  higher better) between predictions and ground truths, must behave like rbf_kernel from sklearn
            alpha: parameter of selection algorithm, 0 for double greedy and 1 for consistent radius
            beta: upper bound on the size of each region as a fraction of total data size
            randomized_sampling:  each round, consider a point with probability self.randomized_sampling in [0,1]: 1 is no sampling
            delta: minimum gain of each region over the prior as raw number of points
            teaching_points: number of teaching points to return
            parallel_processes: run the code in a parallel way with # many processes, 1 is not parallel
        """
        self.data_x = data_x
        self.data_y = data_y
        self.data_y = data_y
        self.ai_preds = ai_preds
        self.kernel = sim_kernel
        self.prior_rejector_preds = prior_rejector_preds
        self.metric_y = metric_y
        self.alpha = alpha
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.delta = delta
        self.randomized_sampling = randomized_sampling
        self.teaching_points = teaching_points
        self.parallel_processes = parallel_processes
        self.threshold_metric = threshold_metric
        print("getting gammas and optimal deferral decisions on teaching set")
        self.get_optimal_deferral()
        # get consistentg
        self.get_optimal_consistent_gammas()

    def get_optimal_deferral(self):
        """
        gets optimal deferral decisions computed emperically
        Return:
            opt_defer: optimal deferral decisions (binary)
        """
        opt_defer_teaching = []
        for ex in range(len(self.data_y)):
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            if score_ai < self.threshold_metric:
                opt_defer_teaching.append(0)
            else:
                opt_defer_teaching.append(1)
        self.opt_defer = np.array(opt_defer_teaching)
        return np.array(opt_defer_teaching)

    def get_optimal_consistent_gammas(self):
        """
        get optimal consistent gammas
        Return:
            optimal_consistent_gammas: array of optimal consistent gamma values
        """
        optimal_consistent_gammas = []
        print("Computing kernel matrix...")
        start = time.time()
        similarities_embeds_all = self.kernel(
            np.asarray(self.data_x), np.asarray(self.data_x)
        )  # kernel matrix
        end = time.time()
        print("Time to compute kernel matrix: ", end - start)
        self.similarities_embeds_all = similarities_embeds_all  # save kernel matrix
        sorted_sims = []
        for i in tqdm(range(len(self.similarities_embeds_all))):
            sorted_sim = sorted(
                [
                    (self.similarities_embeds_all[i][k], k)
                    for k in range(len(self.data_x))
                ],
                key=lambda tup: tup[0],
            )
            sorted_sims.append(np.asarray(sorted_sim))
        self.sorted_sims = sorted_sims

        with tqdm(total=len(self.data_x)) as pbar:
            for i in range(len(self.data_x)):
                # get all similarities
                similarities_embeds = similarities_embeds_all[i]
                opt_defer_ex = self.opt_defer[i]
                opt_gamma = 1
                sorted_sim = self.sorted_sims[
                    i
                ]  # sorted([(similarities_embeds[k], self.opt_defer[k]) for k in range(len(self.data_x))], key=lambda tup: tup[0])
                indicess = list(range(1, len(self.opt_defer)))
                indicess.reverse()
                for k in indicess:
                    if (
                        self.opt_defer[int(sorted_sim[k][1])] == opt_defer_ex
                        and self.opt_defer[int(sorted_sim[k - 1][1])] != opt_defer_ex
                    ):
                        opt_gamma = sorted_sim[k][0]
                        break
                optimal_consistent_gammas.append(opt_gamma)
                pbar.update(1)
        self.optimal_consistent_gammas = np.array(optimal_consistent_gammas)
        return np.array(optimal_consistent_gammas)

    def get_improvement_at_point(self, index_point):
        """
        Gets how much would score improve in terms of metric_y if you add point to the teaching set
        Args:
            index_point: index of point to add
            seen_indices: current predictions of human learner on teaching set

        Return:
            error_improvement: improvement of adding point i for each i in data to the human training set
            found_gamma: optimal gamma found for point
        """
        indicess = list(range(0, len(self.data_x)))
        indicess.reverse()
        if index_point in self.indices_used:
            return -1000, self.optimal_consistent_gammas[index_point]
        # should we consider point?
        coin = random.random()
        if coin >= self.randomized_sampling:
            return -1000, self.optimal_consistent_gammas[index_point]

        sorted_sim = self.sorted_sims[index_point]
        max_improve = -1000
        gamma_value = self.optimal_consistent_gammas[index_point]
        current_improve = 0
        consistency_defer = 0
        so_far = 0
        for j in indicess:
            # max size of neighborhood
            if len(self.data_x) - j >= len(self.data_x) * self.beta_high:
                break
            so_far += 1
            idx = int(sorted_sim[j][1])
            consistency_defer += self.opt_defer[idx]

            if (
                self.opt_defer[index_point] == self.opt_defer[idx]
                and self.current_defer_preds[idx] != self.opt_defer[idx]
            ):
                current_improve += 1
            elif (
                self.opt_defer[index_point] != self.opt_defer[idx]
                and self.current_defer_preds[idx] == self.opt_defer[idx]
            ):
                current_improve -= 1
            else:
                current_improve += 0

            if current_improve >= max_improve:
                feasible = False
                # check constraints for consistency of region

                if self.opt_defer[index_point] == 1:
                    if consistency_defer >= self.alpha * so_far:
                        feasible = True
                else:
                    if consistency_defer <= (1 - self.alpha) * so_far:
                        feasible = True
                if len(self.data_x) - j <= len(self.data_x) * self.beta_low:
                    feasible = False
                if feasible:
                    max_improve = current_improve
                    gamma_value = min(
                        self.optimal_consistent_gammas[index_point], sorted_sim[j][0]
                    )
        return max_improve, gamma_value

    def get_improvement_defer_greedy(self):
        """
        Gets how much would score improve in terms of metric_y if you add each point and optimize radius to the teaching set
        Assumption: assumes metric_y over all data is decomposed as the average of metric_y for each data point e.g. 0-1 loss
        Relaxation: instead of simulating human learner, we assume if point added to the human's set, human will follow the points optimal decision
        Args:
            current_defer_preds: current predictions of human learner on teaching set
            seen_indices: current predictions of human learner on teaching set

        Return:
            error_improvements: improvement of adding point i for each i in data to the human training set
            found_gammas: optimal gammas found for each point
        """
        error_improvements = []
        found_gammas = []

        if self.parallel_processes == 1:
            for i in range(len(self.data_x)):
                max_improve, gamma_value = self.get_improvement_at_point(i)
                error_improvements.append(max_improve)
                found_gammas.append(gamma_value)
        else:
            # parallelize over points and read only shared ressources
            import multiprocessing
            from multiprocessing import Pool

            # get max available processes
            print("Using ", self.parallel_processes, " processes")
            with Pool(processes=self.parallel_processes) as pool:
                results = pool.map(
                    self.get_improvement_at_point, [i for i in range(len(self.data_x))]
                )
                for result in results:
                    error_improvements.append(result[0])
                    found_gammas.append(result[1])

        return error_improvements, found_gammas

    def teach_doublegreedy(self, to_print=False, plotting_interval=2):
        """
        our double greedy selection algorithm, updates human learner
        returns:
            errors: training errors after adding each teaching point
            indices_used: indices used for teaching
        """

        errors = []
        self.indices_used = []
        self.found_gammas = []
        self.current_defer_preds = copy.deepcopy(self.prior_rejector_preds)
        self.teaching_set = []
        plotting_interval = plotting_interval  # plotting interval
        for itt in tqdm(range(self.teaching_points)):
            best_index = -1
            # get improvements for each point if added
            error_improvements, best_gammas = self.get_improvement_defer_greedy()
            # pick best point and add it
            best_index = np.argmax(error_improvements)
            ex_embed = self.data_x[best_index]
            ex_label = self.opt_defer[best_index]
            gamma = best_gammas[best_index]
            if to_print:
                print(f"got improvements with max {max(error_improvements)}")
            # check if improvement is more than minimum gain:
            if max(error_improvements) <= self.delta:
                print(f"improvement is too small, stopping teaching proess")
                return

            self.indices_used.append(best_index)  # add found element to set used
            self.found_gammas.append(gamma)

            # add to teaching set
            self.teaching_set.append((best_index, ex_embed, ex_label, gamma))
            # update current defer preds

            indicess = list(range(0, len(self.data_x)))
            indicess.reverse()
            sorted_sim = self.sorted_sims[best_index]
            for j in indicess:
                if sorted_sim[j][0] <= gamma:
                    break
                idx = int(sorted_sim[j][1])
                self.current_defer_preds[idx] = ex_label

        return

    def get_teaching_examples(self, to_print=False, plot_interval=2):
        """obtains teaching points. Currently only implemented for consistent strategy
        Args:
            to_print: display details of teaching process
            plot_interval: how often to plot results
        Return:
            teaching_x: 2d numpy array of teaching points features
            teaching_indices: indices of the teaching points in self.data_x
            teaching_gammas: 1d numpy of gamma values used
            teaching_labels: 1d array of deferral labels where 1 signifies defer to AI and 0 signifies don't defer to AI

        """

        print("starting the teaching process with greedy radius ...")
        self.teach_doublegreedy(to_print, plot_interval)
        print("got teaching points")
        return

    def get_region_labels(self, data_x, ai_info=None):
        region_labels = np.zeros(len(self.data_x))
        region_label = 1
        for teach_point in self.teaching_set:
            # get similarity to teaching point
            sims = self.kernel(np.asarray([teach_point[1]]), np.asarray(self.data_x))
            in_region = 0
            for i in range(len(sims[0])):
                if sims[0][i] >= teach_point[3]:
                    region_labels[i] = region_label
                    in_region += 1
            region_label += 1
        return region_labels

    def get_data_regions(self):
        data_regions = []
        for teach_point in self.teaching_set:
            # get set of indices of data_x that are within radius of teaching point
            indices = []
            sorted_sim = self.sorted_sims[teach_point[0]]
            for j in range(len(sorted_sim)):
                if sorted_sim[j][0] >= teach_point[3]:
                    indices.append(int(sorted_sim[j][1]))
            indices = np.array(indices)
            # get distr of labels in this region
            labels = self.data_y[indices]
            ai_preds = self.ai_preds[indices]
            # get distribution of labels
            # label_distr = np.bincount(labels)
            # ai_distr = np.bincount(ai_preds)
            # hum_distr = np.bincount(hum_preds)
            data_region = DataRegionModelError(indices, "", teach_point[2], [], [])
            data_regions.append(data_region)
        # get induced cluster labels for each datapoint
        cluster_labels = np.zeros(len(self.data_x))
        for i in range(len(self.data_x)):
            for j in range(len(data_regions)):
                if i in data_regions[j].indices:
                    cluster_labels[i] = j + 1
        return data_regions, cluster_labels
