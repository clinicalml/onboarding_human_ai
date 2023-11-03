
# NEED SKLEARN 0.24.2
# This is edited from the official Domino repo so that it can run
# the version on the site has a few issues with current versions of meerkat, this works easily
# uitls 
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce, wraps
from inspect import getcallargs
from typing import Collection, Mapping
import pandas as pd
import torch
import numpy as np
from typing import List

import meerkat as mk

import warnings
from functools import wraps
from typing import Union

import meerkat as mk
import numpy as np
import sklearn.cluster as cluster
from scipy import linalg
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import _check_X, check_random_state
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_spherical,
    _estimate_gaussian_covariances_tied,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union

import meerkat as mk
import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator





def unpack_args(data: mk.DataPanel, *args):
    if any(map(lambda x: isinstance(x, str), args)) and data is None:
        raise ValueError("If args are strings, `data` must be provided.")

    new_args = []
    for arg in args:
        if isinstance(arg, str):
            arg = data[arg]
        elif isinstance(arg, mk.AbstractColumn):
            # this is necessary because torch.tensor() of a NumpyArrayColumn is very
            # slow and I don't want implementers to have to deal with casing on this
            arg = arg.data
        new_args.append(arg)
    return new_args


def convert_to_numpy(*args):
    """Convert Torch tensors and Pandas Series to numpy arrays."""
    new_args = []
    for arg in args:
        if torch.is_tensor(arg):
            new_args.append(arg.numpy())
        elif isinstance(arg, pd.Series):
            new_args.append(arg.values)
        elif isinstance(arg, List):
            new_args.append(np.array(arg))
        else:
            new_args.append(arg)

    return tuple(new_args)

def convert_to_torch(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, (np.ndarray, pd.Series, List)):
            new_args.append(torch.tensor(arg))
        else:
            new_args.append(arg)
        
    return tuple(new_args)

def nested_getattr(obj, attr, *args):
    """Get a nested property from an object.
    # noqa: E501
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


@dataclass
class VariableColumn:
    variable_name: str

    def resolve(self, args_dict: dict):
        path = self.variable_name.split(".")
        obj = args_dict[path[0]]
        if len(path) > 1:
            return nested_getattr(obj, ".".join(path[1:]))
        return obj


def requires_columns(dp_arg: str, columns: Collection[str]):
    def _requires(fn: callable):
        @wraps(fn)
        def _wrapper(*args, aliases: Mapping[str, str] = None, **kwargs):
            args_dict = getcallargs(fn, *args, **kwargs)
            if "kwargs" in args_dict:
                args_dict.update(args_dict.pop("kwargs"))
            dp = args_dict[dp_arg]
            if aliases is not None:
                dp = dp.view()
                for column, alias in aliases.items():
                    dp[column] = dp[alias]

            # resolve variable columns
            resolved_cols = [
                (col.resolve(args_dict) if isinstance(col, VariableColumn) else col)
                for col in columns
            ]

            missing_cols = [col for col in resolved_cols if col not in dp]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"DataPanel passed to `{fn.__qualname__}` at argument `{dp_arg}` "
                    f"is missing required columns `{missing_cols}`."
                )
            args_dict[dp_arg] = dp
            return fn(**args_dict)

        return _wrapper

    return _requires


@dataclass
class Config:
    pass


class Slicer(ABC, BaseEstimator):
    def __init__(self, n_slices: int):
        super().__init__()

        self.config = Config()
        self.config.n_slices = n_slices

    @abstractmethod
    def fit(
        self,
        model: nn.Module = None,
        data_dp: mk.DataPanel = None,
    ) -> Slicer:
        """
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """

        """
        raise NotImplementedError()

    def get_params(self) -> Dict[str, Any]:
        """

        """
        return self.config.__dict__

    def set_params(self, **params):
        raise ValueError(
            f"Slicer of type {self.__class__.__name__} does not support `set_params`."
        )

    def to(self, device: Union[str, int]):

        if device != "cpu":
            raise ValueError(f"Slicer of type {type(self)} does not support GPU.")
        # by default this is a no-op, but subclasses can override

class MixtureSlicer(Slicer):

    r"""
    Slice Discovery based on the Domino Mixture Model.

    Discover slices by jointly modeling a mixture of input embeddings (e.g. activations
    from a trained model), class labels, and model predictions. This encourages slices
    that are homogeneous with respect to error type (e.g. all false positives).

    Examples
    --------
    Suppose you've trained a model and stored its predictions on a dataset in
    a `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_ with columns
    "emb", "target", and "pred_probs". After loading the DataPanel, you can discover
    underperforming slices of the validation dataset with the following:

    .. code-block:: python

        from domino import MixtureSlicer
        dp = ...  # Load dataset into a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        domino = MixtureSlicer()
        domino.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["domino_slices"] = domino.predict(
            data=test_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


    Args:
        n_slices (int, optional): The number of slices to discover.
            Defaults to 5.
        covariance_type (str, optional): The type of covariance parameter
            :math:`\mathbf{\Sigma}` to use. Same as in sklearn.mixture.GaussianMixture.
            Defaults to "diag", which is recommended.
        n_pca_components (Union[int, None], optional): The number of PCA components
            to use. If ``None``, then no PCA is performed. Defaults to 128.
        n_mixture_components (int, optional): The number of clusters in the mixture
            model, :math:`\bar{k}`. This differs from ``n_slices`` in that the
            ``MixtureSlicer`` only returns the top ``n_slices`` with the highest error rate
            of the ``n_mixture_components``. Defaults to 25.
        y_log_likelihood_weight (float, optional): The weight :math:`\gamma` applied to
            the :math:`P(Y=y_{i} | S=s)` term in the log likelihood during the E-step.
            Defaults to 1.
        y_hat_log_likelihood_weight (float, optional): The weight :math:`\hat{\gamma}`
            applied to the :math:`P(\hat{Y} = h_\theta(x_i) | S=s)` term in the log
            likelihood during the E-step. Defaults to 1.
        max_iter (int, optional): The maximum number of iterations to run. Defaults
            to 100.
        init_params (str, optional): The initialization method to use. Options are
            the same as in sklearn.mixture.GaussianMixture plus one addition,
            "confusion". If "confusion",  the clusters are initialized such that almost
            all of the examples in a cluster come from same cell in the confusion
            matrix. See Notes below for more details. Defaults to "confusion".
        confusion_noise (float, optional): Only used if ``init_params="confusion"``.
            The scale of noise added to the confusion matrix initialization. See notes
            below for more details.
            Defaults to 0.001.
        random_state (Union[int, None], optional): The random seed to use when
            initializing  the parameters.

  
    """

    def __init__(
        self,
        n_slices: int = 5,
        covariance_type: str = "diag",
        n_pca_components: Union[int, None] = 128,
        n_mixture_components: int = 25,
        y_log_likelihood_weight: float = 1,
        y_hat_log_likelihood_weight: float = 1,
        max_iter: int = 100,
        init_params: str = "confusion",
        confusion_noise: float = 1e-3,
        random_state: int = None,
        pbar: bool = True,
    ):
        super().__init__(n_slices=n_slices)

        self.config.covariance_type = covariance_type
        self.config.n_pca_components = n_pca_components
        self.config.n_mixture_components = n_mixture_components
        self.config.init_params = init_params
        self.config.confusion_noise = confusion_noise
        self.config.y_log_likelihood_weight = y_log_likelihood_weight
        self.config.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.config.max_iter = max_iter

        if self.config.n_pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=self.config.n_pca_components)

        self.mm = DominoMixture(
            n_components=self.config.n_mixture_components,
            y_log_likelihood_weight=self.config.y_log_likelihood_weight,
            y_hat_log_likelihood_weight=self.config.y_hat_log_likelihood_weight,
            covariance_type=self.config.covariance_type,
            init_params=self.config.init_params,
            max_iter=self.config.max_iter,
            confusion_noise=self.config.confusion_noise,
            random_state=random_state,
            pbar=pbar,
        )

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = None,
    ) -> MixtureSlicer:
        """
        Fit the mixture model to data.

        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".

        Returns:
            MixtureSlicer: Returns a fit instance of MixtureSlicer.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        if self.pca is not None:
            self.pca.fit(X=embeddings)
            embeddings = self.pca.transform(X=embeddings)
        
        self.mm.fit(X=embeddings, y=targets, y_hat=pred_probs)

        self.slice_cluster_indices = (
            -np.abs((self.mm.y_hat_probs - self.mm.y_probs).max(axis=1))
        ).argsort()[: self.config.n_slices]
        return self

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = "losses",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.


        .. caution::
            Must call ``MixtureSlicer.fit`` prior to calling ``MixtureSlicer.predict``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): Ignored. 

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
        )
        preds = np.zeros_like(probs, dtype=np.int32)
        preds[np.arange(preds.shape[0]), probs.argmax(axis=-1)] = 1
        return preds

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = "loss"
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.

        .. caution::
            Must call ``MixtureSlicer.fit`` prior to calling
            ``MixtureSlicer.predict_proba``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".
            losses (Union[str, np.ndarray], optional): Ignored.

        Returns:
            np.ndarray: A ``np.ndarray`` of shape (n_samples, n_slices) where values in
                are in range [0,1] and rows sum to 1.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        if self.pca is not None:
            embeddings = self.pca.transform(X=embeddings)

        clusters = self.mm.predict_proba(embeddings, y=targets, y_hat=pred_probs)

        return clusters[:, self.slice_cluster_indices]


class DominoMixture(GaussianMixture):
    @wraps(GaussianMixture.__init__)
    def __init__(
        self,
        *args,
        y_log_likelihood_weight: float = 1,
        y_hat_log_likelihood_weight: float = 1,
        confusion_noise: float = 1e-3,
        pbar: bool = True,
        **kwargs,
    ):
        self.y_log_likelihood_weight = y_log_likelihood_weight
        self.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.confusion_noise = confusion_noise
        self.pbar = pbar
        super().__init__(*args, **kwargs)
        self.reg_covar = 1e-6 # causes errors
        
    def _initialize_parameters(self, X, y, y_hat, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp = np.array(resp, dtype='float')
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "confusion":
            num_classes = y.shape[-1]
            if self.n_components < num_classes**2:
                raise ValueError(
                    "Can't use 'init_params=\"confusion\"' when "
                    "`n_components` < `num_classes **2`"
                )
            resp = np.matmul(y[:, :, np.newaxis], y_hat[:, np.newaxis, :]).reshape(
                len(y), -1
            )
            resp = np.array(resp, dtype='float')
            resp = np.concatenate(
                [resp]
                * (
                    int(self.n_components / (num_classes**2))
                    + (self.n_components % (num_classes**2) > 0)
                ),
                axis=1,
            )[:, : self.n_components]
            resp /= resp.sum(axis=1)[:, np.newaxis]

            resp += (
                random_state.rand(n_samples, self.n_components) * self.confusion_noise
            )
            resp /= resp.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self._initialize(X, y, y_hat, resp)

    def _initialize(self, X, y, y_hat, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances, y_probs, y_hat_probs = _estimate_parameters(
            X, y, y_hat, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        self.y_probs, self.y_hat_probs = y_probs, y_hat_probs
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = self.precisions_init

    def fit(self, X, y, y_hat):

        self.fit_predict(X, y, y_hat)
        return self

    def _preprocess_ys(self, y: np.ndarray = None, y_hat: np.ndarray = None):
        if y is not None:
            # we want to support continuous binary labels as well
            if y.dtype == np.dtype(int):
                y = label_binarize(y, classes=np.arange(np.max(y) + 1))
            if y.ndim == 1:
                y = y[:, np.newaxis]
            if y.shape[-1] == 1:
                # binary targets transform to a column vector with label_binarize
                y = np.array([1 - y[:, 0], y[:, 0]]).T
        if y_hat is not None:
            if len(y_hat.shape) == 1:
                y_hat = np.array([1 - y_hat, y_hat]).T
        return y, y_hat

    def fit_predict(self, X, y, y_hat):
        y, y_hat = self._preprocess_ys(y, y_hat)

        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_n_features(X, reset=True)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        best_params = None
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, y, y_hat, random_state)

            lower_bound = -np.infty if do_init else self.lower_bound_

            for n_iter in tqdm(
                range(1, self.max_iter + 1), colour="#f17a4a", disable=not self.pbar
            ):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X, y, y_hat)
                if np.isnan(log_resp).any():
                    print("NANN")
                    import pdb; pdb.set_trace()

                self._m_step(X, y, y_hat, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    print(f'converged at {n_iter}')
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter
        print()
        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        if best_params is None:
            self._initialize_parameters(X, y, y_hat, random_state)
        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, y, y_hat)

        return log_resp.argmax(axis=1)

    def predict_proba(
        self, X: np.ndarray, y: np.ndarray = None, y_hat: np.ndarray = None
    ):
        y, y_hat = self._preprocess_ys(y, y_hat)

        check_is_fitted(self)
        X = _check_X(X, None, self.means_.shape[1])
        _, log_resp = self._estimate_log_prob_resp(X, y, y_hat)
        return np.exp(log_resp)

    def _m_step(self, X, y, y_hat, log_resp):
        """M step.
        """
        resp = np.exp(log_resp)
        n_samples, _ = X.shape
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
        ) = _estimate_parameters(
            X, y, y_hat, resp, self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _e_step(self, X, y, y_hat):
        """E step.

        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, y, y_hat)
        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, X, y=None, y_hat=None):
        """Estimate log probabilities and responsibilities for each sample.

        """
        weighted_log_prob = self._estimate_weighted_log_prob(X, y, y_hat)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X, y=None, y_hat=None):
        log_prob = self._estimate_log_prob(X) + self._estimate_log_weights()

        if y is not None:
            log_prob += self._estimate_y_log_prob(y) * self.y_log_likelihood_weight

        if y_hat is not None:
            log_prob += (
                self._estimate_y_hat_log_prob(y_hat) * self.y_hat_log_likelihood_weight
            )

        return log_prob

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_**2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        return super()._n_parameters() + 2 * self.n_components

    def _estimate_y_log_prob(self, y):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        y: array-like of shape (n_samples, n_classes)

        y_hat: array-like of shpae (n_samples, n_classes)
        """
        # add epsilon to avoid "RuntimeWarning: divide by zero encountered in log"
        return np.log(np.dot(y, self.y_probs.T) + np.finfo(self.y_probs.dtype).eps)

    def _estimate_y_hat_log_prob(self, y_hat):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        y: array-like of shape (n_samples, n_classes)

        y_hat: array-like of shpae (n_samples, n_classes)
        """
        # add epsilon to avoid "RuntimeWarning: divide by zero encountered in log"
        if (np.dot(y_hat, self.y_hat_probs.T) + np.finfo(self.y_hat_probs.dtype).eps < 0).any():
            import pdb; pdb.set_trace()
        
        return np.log(
            np.dot(y_hat, self.y_hat_probs.T) + np.finfo(self.y_hat_probs.dtype).eps
        )


def _estimate_parameters(X, y, y_hat, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # (n_components, )
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)

    y_probs = np.dot(resp.T, y) / nk[:, np.newaxis]  # (n_components, n_classes)
    y_hat_probs = np.dot(resp.T, y_hat) / nk[:, np.newaxis]  # (n_components, n_classes)

    return nk, means, covariances, y_probs, y_hat_probs

DominoSlicer = MixtureSlicer