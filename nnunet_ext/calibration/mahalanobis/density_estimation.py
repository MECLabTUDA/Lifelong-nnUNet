# ------------------------------------------------------------------------------
# Multivariate density estimators.
# ------------------------------------------------------------------------------

import numpy as np
from sklearn.covariance import EmpiricalCovariance

class DensityEstimator():
    def __init__(self):
        self.estimator = None

    def fit(self, np_data):
        r"""np_data should be a numpy array with the shape (#samples, dims)
        """
        self.mean = np.mean(np_data, axis=0)
        np_data_div = np_data - self.mean
        self.estimator.fit(np_data_div)

    def get_score(self, sample):
        r"""sample should be a numpy array with the shape dims"""
        raise NotImplementedError

class GaussianDensityEstimator(DensityEstimator):
    def __init__(self):
        self.estimator = EmpiricalCovariance(assume_centered=False)

    def get_score(self, sample):
        sample = sample-self.mean
        if len(sample.shape) == 1:
            sample = np.expand_dims(sample, axis=0)
        return self.estimator.score(sample)

    def get_mahalanobis(self, sample):
        sample = sample-self.mean
        if len(sample.shape) == 1:
            sample = np.expand_dims(sample, axis=0)
        distance = self.estimator.mahalanobis(sample)
        return np.mean(distance)
