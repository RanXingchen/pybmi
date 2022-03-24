import numpy as np
import pybmi as bmi

from scipy.stats import ttest_ind


def feature_selection(features: np.ndarray, labels: np.ndarray,
                      p: float = 0.01, method: str = None):
    """
    Selecting the features with significant difference with different label.

    Parameters
    ----------
    features : ndarray
        The object features which have shape [L, N] wanted to do feature
        selection. In which L means the number of samples, and N means the
        number of features.
    labels : ndarray
        The labels indicate different state for the features. Labels should
        have be a one dimensional vector with length of L.
    p : float, optional
        The significant level.
    method : str, optional
        The statistic method used to compute the significance. The option
        can be one of: 'welch',
        Default is None, means that if the labels only have 2 state, using
        '' as the method; otherwise, use welch's ANOVA as the method.
    """
    states = np.unique(labels)

    # Initialize the p values.
    sigs = np.zeros((features.shape[-1],))
    # Check the number of states, and do statistics test.
    if len(states) < 2:
        return features
    elif len(states) == 2:
        x0 = features[labels == states[0]]
        x1 = features[labels == states[1]]
        if method is None or method == 'welch':
            _, sigs = ttest_ind(x0, x1, equal_var=False, axis=0)
        # TODO: Add more methods.
    else:
        if method is None or method == 'welch':
            for i, feature in enumerate(features.T):
                sigs[i] = bmi.stats.wanova1(feature, labels)
        # TODO: Add more methods.

    # Selecting the features which had p value smaller than threshold.
    indices = sigs < p
    return features[:, indices], sigs
