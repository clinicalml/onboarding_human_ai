import sys
import torch
from sklearn_extra.cluster import KMedoids
import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import random
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
def squared_distance(x,y):
    return torch.sum((x-y)*(x-y), dim = 1)

def requires_grad(model, condition):
    """Changes all parameters in model to require_grad based oncondition"""
    for p in model.parameters():
        p.requires_grad = condition


class TeachingPoint(torch.nn.Module):
    def __init__(self, teacher_initialize, gamma_initialize, defer_label):
        super().__init__()
        self.x_teach = torch.nn.Parameter(teacher_initialize.detach().clone(), )
        self.gamma = torch.nn.Parameter(gamma_initialize.detach().clone())
        self.defer_label = defer_label
        self.w = torch.nn.Parameter(torch.ones_like(self.x_teach))
        # ablation gradients
        #self.x_teach.requires_grad = False
        #self.w.requires_grad = False
        #self.gamma.requires_grad = False
    def forward(self):
        return self.x_teach, self.gamma

class IntegrAIDiscover:
    '''
    Class to perform the discovery algorithm for the IntegrAI method
    Has two modes: in single mode, it compares the loss of a model to a threshold, in double mode, it compares the loss of two models
    '''
    def __init__(self,
                data_x,
                model1_losses,
                model2_losses = None,
                mode = 'single',
                teaching_points = 10,
                beta_high = 0.05,
                beta_low = 0.001,
                device = 'cpu'):
            """Init function.
            Args:
                data_x: 2d numpy array of the features
                model1_losses: 1d array of floats, loss of the first model on each data point (lower is better) - human
                mode: (string) 'single' (model error) or 'double' (compare two models)
                model2_losses: 1d array of floats,loss of the second model on each data point (lower is better) - ai
                beta_high: upper bound on the size of each region as a fraction of total data size  (strictly enforced)
                beta_low: lower bound on the size of each region as a fraction of total data size (strictly enforced)
                teaching_points: number of teaching points to return
                device: cpu or cuda

            If model1_loss lower than model2_loss then optimal is 0, else 1
            """
            self.data_x = torch.FloatTensor(data_x)
            self.norm_cst =  max(torch.norm(self.data_x, dim = 1).reshape(-1,1))[0]
            self.data_x = self.data_x / self.norm_cst
            self.model1_losses = model1_losses
            self.mode = mode
            if self.mode  not in ['single', 'double']:
                raise ValueError('mode must be single or double')
            if self.mode == 'double':
                if model2_losses is None:
                    raise ValueError('model2_loss must be provided for mode double')
            self.model2_losses = model2_losses
            if beta_high < beta_low:
                raise ValueError('beta_high must be greater than beta_low')
            self.beta_high = beta_high
            self.beta_low = beta_low
            self.teaching_points = teaching_points
            self.device = device

            # Parameters for the discovery algorithm
            self.beta_high_act = beta_high
            self.beta_low_act = beta_low
            # parameter alpha sets consistency of region
            self.alpha = 0.5
            self.delta = 1
            self.lr = 0.01
            self.epochs = 2000
            self.initialization_epochs = 10
            self.large_C = 20
            self.kmeans_nclusters = min(max(100, self.teaching_points), len(self.data_x))
            self.initialization_restarts = max(1, self.kmeans_nclusters)
            self.cutoff_point = int(self.beta_high_act * len(self.data_x))
            self.teaching_set = []

    def get_optimal_deferral(self):
        '''
        gets optimal deferral decisions computed empirically
        0: if model1 is strictly better than model2
        1: if model2 is better than model1
        Return:
            opt_defer: optimal deferral decisions (binary)
        '''
        opt_defer_teaching = []
        for ex in range(len(self.data_x)):
            if  self.model1_losses[ex] < self.model2_losses[ex]:
                opt_defer_teaching.append(0)
            else:
                opt_defer_teaching.append(1)
        self.opt_defer = np.array(opt_defer_teaching)
        return np.array(opt_defer_teaching)


    def update_defer_preds(self):
        """
        Updates the defer predictions based on the current teaching set
        Returns:
            defer_preds: updated defer predictions
        """
        defer_preds = copy.deepcopy(self.prior_rejector_preds)
        # for each point in teaching set, update the defer preds
        with torch.no_grad():
            for tp in self.teaching_set:
                # compute distances
                all_dists = squared_distance(tp.w * self.data_x, tp.w * tp.x_teach)
                in_region = all_dists < tp.gamma
                for ex in range(len(self.data_x)):
                    if in_region[ex] == 1:
                        defer_preds[ex] = tp.defer_label
                # update defer preds
        return defer_preds


    def compute_loss_defer(self, defer_preds):
        """
        Args:
            defer_preds: defer predictions
        Returns:
            loss: loss of the defer predictions
        """
        loss = 0
        for ex in range(len(self.data_x)):
            if defer_preds[ex] == 1:
                loss += self.model2_losses[ex]
            else:
                loss += self.model1_losses[ex]
        return loss/len(self.data_x)

    def optimize_teaching_point(self, teaching_point):
        """
        subroutine to optimize a teaching point used in the fit method
        Args:
            teaching_point: teaching point to optimize
        Returns:
            teaching_point: optimized teaching point
            sum_gains: sum of gains in the region
            points_in_region: number of points in the region
            consistency_in_region: consistency in the region
        """
        # STEP 0: get gain vector and agreement with optimal deferal predictions
        defer_label = teaching_point.defer_label
        # compute gain given label
        gain_vector = []
        agreement_with_opt = torch.tensor(torch.zeros(len(self.data_x))).to(self.device)
        for ex in range(len(self.data_x)):
            model1_loss = self.model1_losses[ex]
            model2_loss = self.model2_losses[ex]
            if defer_label == self.current_defer_preds[ex]:
                # no gain since we already agree with the current defer preds
                gain_vector.append(0.0)
            else:
                if defer_label == 1:
                    gain_vector.append( model1_loss - model2_loss * 1.0)
                else:
                    gain_vector.append( model2_loss - model1_loss *1.0)

            gain_defer_label = (2*defer_label-1) * (model1_loss - model2_loss * 1.0)
            gain_opt_label = (2*self.opt_defer[ex]-1) * (model1_loss - model2_loss * 1.0)
            if gain_opt_label == gain_defer_label:
                agreement_with_opt[ex] = 1.0
            else:
                agreement_with_opt[ex] = 0.0


        gain_vector = torch.tensor(gain_vector).to(self.device)
        teaching_optim = torch.optim.AdamW(list(teaching_point.parameters()), lr=self.lr)

        # STEP 1: find best initialization for teaching point
        sampled_initial_set = random.sample(list(range(len(self.initial_teaching_points))), self.initialization_restarts)
        best_loss = 1000000
        best_index = 0
        for i in range(self.initialization_restarts):
            teaching_point = TeachingPoint(torch.FloatTensor([self.initial_teaching_points[sampled_initial_set[i]]]), torch.tensor(-0.0), defer_label ).to(self.device)
            teaching_optim = torch.optim.AdamW(list(teaching_point.parameters()), lr=self.lr)
            for epoch in range(self.initialization_epochs):
                loss = 0
                teaching_point.zero_grad()
                x_teach, gamma = teaching_point()
                requires_grad(teaching_point, True)
                all_dists = squared_distance(teaching_point.w * self.data_x, teaching_point.w * x_teach)
                all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
                loss_region_size_over = F.relu((torch.sum(all_dists) - self.beta_high * len(self.data_x)))
                loss_region_size_under = -F.relu((torch.sum(all_dists) - self.beta_low * len(self.data_x)))/len(self.data_x) * 10
                loss_point_wise = all_dists * gain_vector * -1
                loss_consistency = F.relu( (self.alpha * torch.sum(all_dists) - torch.sum(all_dists * agreement_with_opt) ))
                loss += torch.sum(loss_point_wise)
                loss += loss_region_size_over
                loss += loss_region_size_under
                loss += loss_consistency
                loss.backward()
                teaching_optim.step()
            if loss < best_loss:
                best_loss = loss
                best_index = i
        logging.info(f'Initialization is done: best loss is {best_loss} and best index is {best_index}')

        # STEP 2: optimize teaching point given best initialization
        teaching_point = TeachingPoint(torch.FloatTensor([self.initial_teaching_points[sampled_initial_set[best_index]]]), torch.tensor(-0.0), defer_label ).to(self.device)
        teaching_optim = torch.optim.AdamW(list(teaching_point.parameters()), lr=self.lr)
        scheduler = ReduceLROnPlateau(teaching_optim, factor = 0.9, patience = 50, min_lr = 0.0001)


        for epoch in range(self.epochs):
            loss = 0
            teaching_point.zero_grad()
            x_teach, gamma = teaching_point()
            requires_grad(teaching_point, True)
            all_dists = squared_distance(teaching_point.w * self.data_x, teaching_point.w * x_teach)
            all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
            loss_region_size_over = F.relu((torch.sum(all_dists) - self.beta_high * len(self.data_x)))
            loss_region_size_under = F.relu((-torch.sum(all_dists) + self.beta_low * len(self.data_x)))
            loss_point_wise = all_dists * gain_vector * -1
            loss_consistency = F.relu( (self.alpha * torch.sum(all_dists) - torch.sum(all_dists * agreement_with_opt) ))
            loss += torch.sum(loss_point_wise)
            loss += loss_region_size_over
            loss += loss_region_size_under
            loss += loss_consistency
            loss.backward()
            teaching_optim.step()
            scheduler.step(loss)
        logging.info(f' epoch {epoch} loss is {loss.item()} lr {teaching_optim.param_groups[0]["lr"]} sched {scheduler.optimizer.param_groups[0]["lr"]}')
        logging.info(f'loss region size {loss_region_size_over} losscons {loss_consistency}')

        # STEP 3: get improvement in gain total, and force region to be of size beta_high
        points_in_region = 0
        actual_gain_vector = []
        consistency_in_region = 0
        with torch.no_grad():
            teaching_point.zero_grad()
            x_teach, gamma = teaching_point()
            all_dists = squared_distance(teaching_point.w * self.data_x, teaching_point.w * x_teach)
            all_dists_sig = torch.sigmoid(self.large_C *(-all_dists + gamma))
            # count how all_dists_sig are above 0.5
            count_in_region = torch.sum(all_dists_sig >= 0.5)
            cutoff_point = self.cutoff_point
            # if count_in_region is greater than cutoff_point, readjust gamma
            if count_in_region > cutoff_point:
                logging.info(f'count in region {count_in_region} is greater than cutoff point {cutoff_point}, readjusting gamma')
                all_dists_sorted, indices = torch.sort(all_dists)
                dist_cutoff = all_dists_sorted[cutoff_point + 1]
                teaching_point.gamma = torch.nn.Parameter(dist_cutoff)
                gamma = dist_cutoff
            minimum_region_size = int(self.beta_low * len(self.data_x))
            if count_in_region < minimum_region_size:
                logging.info(f'count in region {count_in_region} is less than minimum region size {minimum_region_size}, readjusting gamma')
                all_dists_sorted, indices = torch.sort(all_dists, descending = True)
                dist_cutoff = all_dists_sorted[minimum_region_size + 1]
                teaching_point.gamma = torch.nn.Parameter(dist_cutoff)
                gamma = dist_cutoff

            all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
            for ex in range(len(self.data_x)):
                in_region = all_dists[ex]
                if in_region >= 0.5:
                    actual_gain_vector.append(gain_vector[ex])
                    points_in_region += 1
                    consistency_in_region += agreement_with_opt[ex].item()
            sum_gains = torch.sum(torch.tensor(actual_gain_vector))
            logging.info(f'final loss is {loss} and final gain is {sum_gains.item()} and region size {points_in_region}')
            return teaching_point, sum_gains.item(), points_in_region, consistency_in_region/(points_in_region+1)


    def fit(self, prior, loss_threshold = None, model2_losses = None):
        """
        Args:
            loss_threshold: (only used in "single" mode) threshold on loss to denote if a loss of an example is high or low (binarize the loss). If you are using as your loss metric misclassification error, you should set this to 0.5

            prior: predictions of the prior rejector (1d binary array where 1 is model2 better than model1 and 0 otherwise), can also use:
            "all_1": all 1s prior (model2 better than model1 on all points) or "all_0": all 0s prior (model1 better than model2), or "mix": random mix of 1s and 0s (0.5 probability of 1)
            in mode "single": "all_1" means our prior is that model1 has loss higher than loss_threshold on all points, "all_0" means model1 has loss lower than loss_threshold, "mix" means that we are unsure about the prior
        Returns:
            teaching_set: list of teaching points
        """
        # check if prior is string
        if isinstance(prior, str):
            if prior == 'all_1':
                self.prior_rejector_preds = np.ones(len(self.data_x))
            elif prior == 'all_0':
                self.prior_rejector_preds = np.zeros(len(self.data_x))
            elif prior == 'mix':
                self.prior_rejector_preds = np.random.choice([0, 1], size=(len(self.data_x)), p=[0.5, 0.5])
            else:
                raise ValueError('prior must be all_1, all_0, mix, or a 1d array of predictions')
        else:
            self.prior_rejector_preds = prior

        if self.mode == 'single':
            # must provide loss_threshold
            if loss_threshold is None and model2_losses is None:
                raise ValueError('loss_threshold must be provided for mode single, or provide a reference model2 losses array')
            if model2_losses is not None:
                self.model2_losses = model2_losses
            else:
                self.model2_losses = np.zeros(len(self.data_x))
                self.model2_losses += loss_threshold
                self.loss_threshold = loss_threshold


        self.teaching_set = []
        logging.info("Getting gammas and optimal deferral decisions on teaching set")
        self.get_optimal_deferral()
        loss_optimal = self.compute_loss_defer(self.opt_defer)
        logging.info(f'optimal loss is {loss_optimal}')
        # run kmeans to get initial teaching points
        logging.info("running kmeans to get initial teaching points")
        #kmeans = sklearn.cluster.KMeans(n_clusters=self.kmeans_nclusters, random_state=0).fit(self.data_x)
        kmeans = KMedoids(n_clusters=self.kmeans_nclusters, init='k-medoids++').fit(self.data_x)
        self.initial_teaching_points = kmeans.cluster_centers_
        # shuffle initial teaching points
        self.current_defer_preds = copy.deepcopy(self.prior_rejector_preds)
        random.shuffle(self.initial_teaching_points)
        # start discovery
        teaching_points_real = 0
        for itt in tqdm(range(min(self.teaching_points * 2, len(self.data_x)))):
            # get the next teaching point
            logging.info(f'Getting teaching point {teaching_points_real + 1} at iteration {itt+1}')
            # pick random initial teaching point from initial_teaching_points
            initial_teaching_point = torch.FloatTensor([self.initial_teaching_points[itt]])
            # try for defer_label 1 and 0
            max_gain = -10000
            max_region = 0
            max_consist_reg = 0
            best_teaching_point = None
            for defer_label in [1, 0]:
                # create teaching point object
                teaching_point = TeachingPoint(initial_teaching_point, torch.tensor(-0.0), defer_label ).to(self.device)
                # optimize teaching point
                teaching_point, gain_point, points_in_region, consist_reg = self.optimize_teaching_point(teaching_point)
                if gain_point > max_gain:
                    max_gain = gain_point
                    best_teaching_point = teaching_point
                    max_region = points_in_region
                    max_consist_reg = consist_reg
            logging.info(f'optimized point with defer lablel {defer_label}, gain is {max_gain} and region size {max_region} and consistency {max_consist_reg}')
            # add best teaching point to teaching set
            if max_gain > self.delta:
                self.teaching_set.append(best_teaching_point)
                # update current defer preds
                self.current_defer_preds = self.update_defer_preds()
                teaching_points_real += 1

            else:
                logging.info(f'gain is too low, skipping region found')
            # compute metrics
            current_loss = self.compute_loss_defer(self.current_defer_preds)
            logging.info(f'current loss is {current_loss}')
            if len(self.teaching_set) >= self.teaching_points:
                logging.info(f'DONE TEACHING')
                print(f'Found {len(self.teaching_set)} regions')
                return self.teaching_set
        logging.info(f'DONE TEACHING')
        print(f'Found {len(self.teaching_set)} regions')
        return self.teaching_set

    def get_defer_preds(self, data_x, prior_rejector = None):
        """
        Args:
            data_x: data to get predictions on
            prior_rejector: predictions of the prior rejector
        Returns:
            defer_preds: predictions for each example
        """
        data_x = torch.FloatTensor(data_x)
        data_x = data_x / self.norm_cst
        if prior_rejector is not None:
            defer_preds = np.array(prior_rejector)
        else:
            defer_preds = np.zeros(len(data_x))
        for teaching_point in self.teaching_set:
            x_teach, gamma = teaching_point()
            all_dists = squared_distance(teaching_point.w * data_x, teaching_point.w * x_teach)
            all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
            for ex in range(len(data_x)):
                if all_dists[ex] > 0.5:
                    defer_preds[ex] = teaching_point.defer_label
        return defer_preds




    def get_region_labels(self, data_x):
        """
        Args:
            data_x: data to get predictions on
        Returns:
            region_labels: predictions for each example
        """
        region_labels = np.zeros(len(data_x))
        counter = 1
        data_x = torch.FloatTensor(data_x)
        data_x = data_x /  self.norm_cst
        for teaching_point in self.teaching_set:
            x_teach, gamma = teaching_point()
            all_dists = squared_distance(teaching_point.w *  data_x, teaching_point.w * x_teach)
            all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
            for ex in range(len(data_x)):
                if all_dists[ex] > 0.5:
                    region_labels[ex] = counter
            counter += 1
        return region_labels


    def get_region_labels_probs(self, data_x):
        """
        Args:
            data_x: data to get predictions on
        Returns:
            region_labels: predictions for each example
        """

        region_labels_probs = np.zeros((len(data_x), len(self.teaching_set)))
        counter = 1
        data_x = torch.FloatTensor(data_x)
        data_x = data_x /  self.norm_cst
        for teaching_point in self.teaching_set:
            x_teach, gamma = teaching_point()
            all_dists = squared_distance(teaching_point.w *  data_x, teaching_point.w * x_teach)
            for ex in range(len(data_x)):
                in_region = torch.sigmoid((-all_dists[ex] + gamma))
                region_labels_probs[ex, counter-1] = in_region
            counter += 1
        return region_labels_probs
