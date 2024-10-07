import numpy as np
from scipy.stats import rankdata

# cldice metric from https://github.com/jocpae/clDice/tree/master/cldice_metric

from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def cl_dice_metric(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    else:
        return 'wrong shapes'
    return 2*tprec*tsens/(tprec+tsens)

def fast_bin_auc(actual, predicted, partial=False):
    actual, predicted = actual.flatten(), predicted.flatten()
    if partial:
        n_nonzeros = np.count_nonzero(actual)
        n_zeros = len(actual) - n_nonzeros
        k = min(n_zeros, n_nonzeros)
        predicted = np.concatenate([
            np.sort(predicted[actual == 0])[::-1][:k],
            np.sort(predicted[actual == 1])[::-1][:k]
        ])
        actual = np.concatenate([np.zeros(k), np.ones(k)])

    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0: return 0
    return (np.sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def fast_bin_dice(actual, predicted):
    actual = np.asarray(actual).astype(bool) # bools are way faster to deal with by numpy
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

