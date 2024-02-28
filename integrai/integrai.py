import sys
from integrai_discover import IntegrAIDiscover
from integrai_describe import IntegrAIDescribe
import numpy as np
import torch

class IntegrAI:
    def __init__(
        self,
        embeddings,
        descriptions,
        embedding_func,
        model1_losses,
        model2_losses=None,
        mode="single",
    ):
        """
        This class is used to perform the IntegrAI method
        embeddings: np.array, embedding of the examples (each row is an example)
        descriptions: list of str, the descriptions of the points
        embedding_func: function, the function to get the embedding of the text - given a string, returns a vector embedding, the embedding should live in the same space of the embeddings
        model1_losses: 1d array of floats, loss of the first model on each data point (lower is better)
        model2_losses: 1d array of floats,loss of the second model on each data point (lower is better)
        mode: (string) 'single' (model error) or 'double' (compare two models)
        For arguments of the methods, see IntegrAIDiscover and IntegrAIDescribe
        """
        self.region_finder = IntegrAIDiscover(
            embeddings, model1_losses, model2_losses, mode
        )
        self.region_describer = IntegrAIDescribe(
            descriptions, embeddings, None, None, embedding_func
        )

    def discover_regions(
        self,
        prior,
        loss_threshold=None,
        model2_losses=None,
        number_regions=1,
        max_size=0.05,
        min_size=0.001,
        consistency=0.5,
        min_gain=1,
    ):
        """
        This method is used to discover the regions of the space, calls the region_finder.fit method
        """
        self.region_finder.teaching_points = number_regions
        self.region_finder.beta_high = max_size
        self.region_finder.beta_low = min_size
        self.region_finder.alpha = consistency
        self.region_finder.delta = min_gain
        self.teaching_set = self.region_finder.fit(prior, loss_threshold, model2_losses)
        # update region describer
        region_scores = self.region_finder.get_region_labels_probs(
            self.region_finder.data_x
        )
        region_scores_new = []
        for i in range(len(region_scores)):
            temp = [0]
            for j in range(len(region_scores[i])):
                temp.append(region_scores[i][j])
            region_scores_new.append(np.array(temp))
        region_scores = np.array(region_scores_new)
        self.region_scores = region_scores
        region_labels = self.region_finder.get_region_labels(self.region_finder.data_x)
        self.region_labels = region_labels
        self.region_describer.region_scores = region_scores
        self.region_describer.region_labels = region_labels
        return self.teaching_set

    def describe_region(self, region_index, n_rounds=5, initial_positive_set_size=15, initial_negative_set_size=5, points_to_keep=500):
        """
        This method is used to describe a region of the space, calls the region_describer.describe_region method
        """
        # cannot call this method before calling discover_regions
        if not hasattr(self, "teaching_set"):
            raise ValueError(
                "You must call discover_regions before calling describe_region"
            )
        self.region_describer.n_rounds = n_rounds
        self.region_describer.initial_positive_set_size = initial_positive_set_size
        self.region_describer.initial_negative_set_size = initial_negative_set_size
        self.region_describer.points_to_keep = min(points_to_keep, len(self.region_describer.descriptions))
        return self.region_describer.describe_region(region_index)