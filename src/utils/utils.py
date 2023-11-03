# utilities used

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

kernel = rbf_kernel  # K(.,.)


def loss_01(x, y):
    return 1 - accuracy_score(x, y)


def loss_01_pointwise(labels, responses):
    return [1 if labels[i] != responses[i] else 0 for i in range(len(labels))]


def get_distribution_list(list_x):
    """
    Input: list_x is a list of numbers
    Output: dist is a dictionary where the keys are the numbers in list_x and the values are the probabilities of the numbers in list_x
    """
    dist = {}
    for i in list_x:
        if i in dist:
            dist[i] += 1
        else:
            dist[i] = 1
    for i in dist:
        dist[i] /= len(list_x)
    return dist
