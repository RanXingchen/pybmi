import numpy as np
import warnings

from scipy.stats import ttest_ind

from .statistics import wanova1
from ..utils import check_params


def feature_selection(features: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.01, method: str = 'welch',
                      keep_dims=False):
    """
    Selecting the features with significant difference with different labels.

    Parameters
    ----------
    features : ndarray
        The object features which have shape [L, N] wanted to do feature
        selection. In which L means the number of samples, and N means
        the number of features.
    labels : ndarray
        The labels indicate different state for the features. Labels must
        be a one dimensional vector with length of L.
    p : float, optional
        The significant level.
    method : str, optional
        The statistic method used to compute the significance. The current
        option only 'welch'.
        Default is 'welch', if the labels only have 2 state, using
        welch's t-test, otherwise, use welch's ANOVA.
    keep_dims : bool, optional
        Indicate whether keep the original number of dimensions. If True,
        the features with no testing significance will be replaced to zeros.
        Otherwise, these features are no longer exist in the return results.
        Default: False.
    """
    method = check_params(method, ['welch'], 'method')

    assert features.ndim == 2, \
        'Do not accept features with number of dimensions greater than 2.'
    assert labels.ndim == 1, "Do not accept multi-dimensional labels."

    T = features.shape[-2]  # The length of the features.
    N = features.shape[-1]  # The number of the features.

    assert T == len(labels), \
        "The length of features must equal to the length of states."

    states = np.unique(labels)

    # Check the number of states, and do statistics test.
    if len(states) < 2:
        warnings.warn(
            message=f'The state of the given LABEL only contain {len(states)} '
                    'category. No feature selection applied, the original '
                    'features have been returned.',
            category=type[UserWarning]
        )
        return features
    elif len(states) == 2:
        # Two sample test.
        x0 = features[labels == states[0]]
        x1 = features[labels == states[1]]
        if method == 'welch':
            _, p = ttest_ind(x0, x1, equal_var=False, axis=0)
        # TODO: Add more methods.
    else:
        # Initialize the p values.
        p = np.zeros((N,))

        if method == 'welch':
            for i, feature in enumerate(features.T):
                p[i] = wanova1(feature, labels)
        # TODO: Add more methods.

    # Selecting the features which had p value smaller than threshold.
    indices = p < threshold

    if keep_dims:
        features[:, ~indices] = 0
    else:
        features = features[:, indices]
    return features, p
