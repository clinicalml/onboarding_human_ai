import logging
import os

import numpy as np
import torch

# https://github.com/HazyResearch/domino/blob/main/domino/_describe/generate.py


class DOMINODescribe:
    # not exactly DOMINO as no initial point
    def __init__(self, descriptions, embeddings, cluster_labels):
        self.descriptions = descriptions
        self.cluster_labels = cluster_labels
        self.embeddings = embeddings
        self.descriptions_gen = []

    def describe_region(self, cluster_selected):
        inside_caption_set_idx = []
        for j in range(len(self.cluster_labels)):
            if self.cluster_labels[j] == cluster_selected:
                inside_caption_set_idx.append(j)
        # get average embedding of inside_caption_set_idx
        average_embedding = np.mean(self.embeddings[inside_caption_set_idx], axis=0)
        # get closest caption to average embedding that is in inside_caption_set_idx
        closest_caption_idx = np.argmin(
            np.linalg.norm(
                self.embeddings[inside_caption_set_idx] - average_embedding, axis=1
            )
        )
        # get description
        description = self.descriptions[inside_caption_set_idx[closest_caption_idx]]
        self.descriptions_gen.append(description)
        logging.info(f" Description: {description} ")
        return description
