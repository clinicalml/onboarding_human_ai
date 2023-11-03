import random
import sys

import numpy as np
import torch
import torch.distributions as D

sys.path.append("..")
sys.path.append("../utils")
sys.path.append("./datasets_hai")
from utils.utils import loss_01

from .basedataset import BaseDataset


class MixOfGuassians(BaseDataset):
    """ "
    This class generates a mixture of guassians dataset (d , n, k)
    the label Y is a random half space that splits the data into two groups
    the human and AI predictions are perfect on 1/4 of the guassians, random on 1/4 of the guassians, and 70% correct on the other 1/2 of the guassians
    """

    def __init__(
        self, d, n, k, base_ai_accuracy=0.7, base_human_accuracy=0.7, gmm=None
    ):
        """
        d: dimension of the data
        n: number of samples
        k: number of guassians
        """
        self.d = d
        self.n = n
        self.k = k
        self.k_classes = 2
        self.base_ai_accuracy = base_ai_accuracy
        self.base_human_accuracy = base_human_accuracy
        self.gmm = gmm
        self.data = self.generate_data()

    def get_length():
        return self.n

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

    def generate_data(self):
        if self.gmm is None:
            mix = D.Categorical(
                torch.ones(
                    self.k,
                )
            )
            comp = D.Independent(
                D.Normal(torch.randn(self.k, self.d), torch.rand(self.k, self.d)), 1
            )
            self.gmm = torch.distributions.mixture_same_family.MixtureSameFamily(
                mix, comp
            )
        self.data_x = self.gmm.sample((self.n,))
        which_guass = torch.zeros(self.n)
        for i in range(self.n):
            which_guass[i] = torch.argmax(comp.log_prob(self.data_x[i]))
        which_guass = which_guass.long()
        self.true_regions = np.array(which_guass)

        #  seperate the data into two groups using a linear line such that 50% of the data is in each group
        data_y = np.zeros(self.n)
        proportion_half = 0
        while proportion_half > 0.6 or proportion_half < 0.4:
            line = torch.randn(self.d)
            for i in range(self.n):
                if torch.dot(self.data_x[i], line) > 0:
                    data_y[i] = 1
                else:
                    data_y[i] = 0
            proportion_half = np.mean(data_y)
        self.data_y = data_y.astype(int)
        # let hum_preds be perfect on 1/4 of guassians, random on 1/4 of guassians, and self.base_human_accuracy correct on the other 1/2 of guassians
        hum_perf_clust = random.sample(range(self.k), int(self.k / 4))
        hum_rand_clust = random.sample(range(self.k), int(self.k / 4))
        hum_70_clust = [
            i
            for i in range(self.k)
            if i not in hum_perf_clust and i not in hum_rand_clust
        ]
        hum_preds = np.zeros(self.n)
        for i in range(self.n):
            if which_guass[i] in hum_perf_clust:
                hum_preds[i] = data_y[i]

            elif which_guass[i] in hum_rand_clust:
                hum_preds[i] = np.random.randint(0, 2)
            else:
                if np.random.rand() < self.base_human_accuracy:
                    hum_preds[i] = self.data_y[i]
                else:
                    hum_preds[i] = 1 - self.data_y[i]
        # same for ai_preds
        ai_perf_clust = random.sample(range(self.k), int(self.k / 4))
        ai_rand_clust = random.sample(range(self.k), int(self.k / 4))
        ai_70_clust = [
            i
            for i in range(self.k)
            if i not in ai_perf_clust and i not in ai_rand_clust
        ]
        ai_preds = np.zeros(self.n)
        for i in range(self.n):
            if which_guass[i] in ai_perf_clust:
                ai_preds[i] = data_y[i]
            elif which_guass[i] in ai_rand_clust:
                ai_preds[i] = np.random.randint(0, 2)
            else:
                if np.random.rand() < self.base_ai_accuracy:
                    ai_preds[i] = self.data_y[i]
                else:
                    ai_preds[i] = 1 - self.data_y[i]
        for i in range(self.n):
            if (
                which_guass[i] not in ai_perf_clust
                and which_guass[i] not in ai_rand_clust
                and which_guass[i] not in hum_perf_clust
                and which_guass[i] not in hum_rand_clust
            ):
                self.true_regions[i] = -1
        unique_nonneg_values = np.unique(self.true_regions[self.true_regions != -1])

        # Create a mapping dictionary
        mapping_dict = {
            value: index + 1 for index, value in enumerate(unique_nonneg_values)
        }

        # Map the values in the array using the dictionary, keeping -1 as is
        self.true_regions = np.vectorize(
            lambda val: 0 if val == -1 else mapping_dict[val]
        )(self.true_regions)

        self.data_x = np.array(self.data_x)
        self.hum_preds = hum_preds.astype(int)
        self.ai_preds = ai_preds.astype(int)
        self.data_embs = [self.data_x]
        self.which_embs = [0]
        self.captions = np.array(["" for i in range(self.n)])
        self.caption_embs = np.zeros((self.n, 1))
        self.metric_y = loss_01
        self.get_optimal_deferral()
