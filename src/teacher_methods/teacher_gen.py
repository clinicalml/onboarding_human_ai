import numpy as np
import random
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append("../utils")
sys.path.append("./teacher_methods")
import torch 
import sklearn.cluster
from sklearn_extra.cluster import KMedoids
import copy
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
import random
import math
import copy
import pickle
import logging
from multiprocessing import Pool
import multiprocessing
from utils.metrics_hai import compute_metrics
from utils.utils import *
from .base_teacher import * 
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    
class TeacherGenerative(BaseTeacher):
    """ 
    Generative teaching algorithm greedy one by one

    """
    def __init__(self,
                data_x,
                data_y,
                hum_preds,
                ai_preds,
                prior_rejector_preds,
                metric_y,
                teaching_points = 10,
                alpha = 0.0,
                beta_high = 0.05,
                beta_low = 0.001,
                delta = 2,
                device = 'cpu'):
            """Init function.
            Args:
                data_x: 2d numpy array of the features
                data_y: 1d numpy array of labels
                hum_preds:  1d array of the human predictions 
                ai_preds:  1d array of the AI predictions 
                prior_rejector_preds: 1d binary array of the prior rejector preds 
                sim_kernel: function that takes as input two inputs and returns a positive number
                metric_y: metric function (positive,  lower better) between predictions and ground truths, must behave like rbf_kernel from sklearn
                alpha: parameter of selection algorithm, 0 for double greedy and 1 for consistent radius
                beta: upper bound on the size of each region as a fraction of total data size 
                delta: minimum gain of each region over the prior as raw number of points
                teaching_points: number of teaching points to return
                device: cpu or cuda
            """
            self.data_x = torch.FloatTensor(data_x)
            self.norm_cst =  max(torch.norm(self.data_x, dim = 1).reshape(-1,1))[0]
            self.data_x = self.data_x / self.norm_cst 
            self.data_y = data_y
            self.hum_preds = hum_preds
            self.ai_preds = ai_preds
            self.prior_rejector_preds = prior_rejector_preds
            self.metric_y = metric_y
            self.alpha = alpha
            self.beta_high = beta_high
            self.beta_low = beta_low
            self.beta_high_act = beta_high
            self.delta = delta
            self.teaching_points = teaching_points
            self.device = device
            # can pass them as HPs
            self.lr = 0.01
            self.epochs = 2000
            self.initialization_epochs = 10
            self.large_C = 20
            self.kmeans_nclusters = min(max(100, self.teaching_points), len(self.data_y))
            self.initialization_restarts = max(1, self.kmeans_nclusters)
            self.cutoff_point = int(self.beta_high_act * len(self.data_y))
            self.teaching_set = []
    def get_optimal_deferral(self):
        '''
        gets optimal deferral decisions computed emperically
        Return:
            opt_defer: optimal deferral decisions (binary)
        '''
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


    def update_defer_preds(self):
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


    def optimize_teaching_point(self, teaching_point):
        defer_label = teaching_point.defer_label
        # compute gain given label
        gain_vector = []
        for ex in range(len(self.data_x)):
            score_hum = self.metric_y([self.data_y[ex]], [self.hum_preds[ex]])
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            if defer_label == self.current_defer_preds[ex]:
                gain_vector.append(0.0)
            else:
                if defer_label == 1:
                    gain_vector.append( score_hum - score_ai * 1.0)
                else:
                    gain_vector.append( score_ai - score_hum *1.0)
        agreement_with_opt = torch.tensor((self.opt_defer == defer_label) * 1.0).to(self.device)    
        # for classification tasks, the below formulation makes more sense:                    
        for ex in range(len(self.data_x)):
            score_hum = self.metric_y([self.data_y[ex]], [self.hum_preds[ex]])
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            gain_defer_label = (2*defer_label-1) * (score_hum - score_ai * 1.0)
            gain_opt_label = (2*self.opt_defer[ex]-1) * (score_hum - score_ai * 1.0)
            if gain_opt_label == gain_defer_label:
                agreement_with_opt[ex] = 1.0
            else:
                agreement_with_opt[ex] = 0.0
        
        gain_vector = torch.tensor(gain_vector).to(self.device)
        teaching_optim = torch.optim.AdamW(list(teaching_point.parameters()), lr=self.lr)
                                             
        #teaching_point = TeachingPoint(initial_teaching_point, torch.tensor(0.5), defer_label ).to(self.device)
        # sample self.initialization_restarts points from initial_teaching_point set
        
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
                loss_region_size_over = F.relu((torch.sum(all_dists) - self.beta_high * len(self.data_y)))
                loss_region_size_under = -F.relu((torch.sum(all_dists) - self.beta_low * len(self.data_y)))/len(self.data_y) * 10
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
        logging.info(best_loss)
        teaching_point = TeachingPoint(torch.FloatTensor([self.initial_teaching_points[sampled_initial_set[best_index]]]), torch.tensor(-0.0), defer_label ).to(self.device)            
        teaching_optim = torch.optim.AdamW(list(teaching_point.parameters()), lr=self.lr)
        scheduler = ReduceLROnPlateau(teaching_optim, factor = 0.9, patience = 50, min_lr = 0.0001) 

        for epoch in range(self.epochs):
            # no need to batch data
            loss = 0
            teaching_point.zero_grad()
            x_teach, gamma = teaching_point()
            requires_grad(teaching_point, True)
            all_dists = squared_distance(teaching_point.w * self.data_x, teaching_point.w * x_teach)
            all_dists = torch.sigmoid(self.large_C *(-all_dists + gamma))
            loss_region_size_over = F.relu((torch.sum(all_dists) - self.beta_high * len(self.data_y)))
            loss_region_size_under = F.relu((-torch.sum(all_dists) + self.beta_low * len(self.data_y)))
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
        # get improvement in gain total
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
        
            if count_in_region > cutoff_point:
                logging.info(f'count in region {count_in_region} is greater than cutoff point {cutoff_point}, readjusting gamma')
                all_dists_sorted, indices = torch.sort(all_dists)
                dist_cutoff = all_dists_sorted[cutoff_point + 1]
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
        

    def fit(self, to_print = False):
        """ 
        Args:
            to_print: display details of teaching process
        """
        # run algorithm to get examples
        # get optimal deferrall points
        logging.info("getting gammas and optimal deferral decisions on teaching set")
        self.get_optimal_deferral()
        logging.info(f' SET YOUR EXPECTATIONS CORRECTLY, OPTIMAL LOSS GETS YOU:')
        _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, self.opt_defer, self.data_y, self.metric_y, to_print)
        logging.info(metricss['score'])   
        # run kmeans to get initial teaching points
        logging.info("running kmeans to get initial teaching points")
        #kmeans = sklearn.cluster.KMeans(n_clusters=self.kmeans_nclusters, random_state=0).fit(self.data_x)
        kmeans = KMedoids(n_clusters=self.kmeans_nclusters, init='k-medoids++').fit(self.data_x)
        self.initial_teaching_points = kmeans.cluster_centers_
        # shuffle initial teaching points
        self.current_defer_preds = copy.deepcopy(self.prior_rejector_preds)
        random.shuffle(self.initial_teaching_points)

        teaching_points_real = 0
        for itt in tqdm(range(min(self.teaching_points * 2, len(self.data_x)))):
            # get the next teaching point
            logging.info(f' getting teaching point {teaching_points_real + 1} at iteration {itt+1}')
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
            logging.info(f'gain is {max_gain} and region size {max_region} and consistency {max_consist_reg}')
            # add best teaching point to teaching set
            if max_gain > self.delta:
                self.teaching_set.append(best_teaching_point)                
                # update current defer preds
                self.current_defer_preds = self.update_defer_preds()
                teaching_points_real += 1
                
            else:
                logging.info(f' gain is too low, skipping region found')
            _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, self.current_defer_preds, self.data_y, self.metric_y, to_print)
            logging.info(metricss['score'])   
            if len(self.teaching_set) >= self.teaching_points:
                logging.info(f'DONE TEACHING')
                return
            
    def get_defer_preds(self, data_x, prior_rejctor = None, ai_info = None):
        """
        Args:
            data_x: data to get predictions on
            ai_info: not used
        Returns:
            defer_preds: predictions for each example
        """
        data_x = torch.FloatTensor(data_x)
        data_x = data_x / self.norm_cst 
        if prior_rejctor is not None:
            defer_preds = np.array(prior_rejctor)
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




    def get_region_labels(self, data_x, ai_info = None):
        """
        Args:
            data_x: data to get predictions on
            ai_info: not used
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
    

    def get_region_labels_probs(self, data_x, ai_info = None):
        """
        Args:
            data_x: data to get predictions on
            ai_info: not used
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
    