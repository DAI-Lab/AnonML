import pandas as pd
import numpy as np
import pdb
import random
from anonml.aggregator import get_privacy_params, get_rappor_params, \
                                best_perturb_method
from sklearn.model_selection import KFold


def postprocess_histogram_mle(hist, p, q, n, m):
    """
    do a simple MLE estimate on the histogram, same as what RAPPOR does

    hist: np.array() of floats
    """
    hist -= q * n
    hist /= (p - q)

    return hist


def postprocess_histogram_mle_noneg(hist, p, q, n, m):
    """
    MLE technique with some extra fanciness to get rid of negative entries and
    bring down outliers. Not generally as good.

    hist: np.array() of floats
    """
    hist -= q * n
    hist /= (p - q)

    mean = float(n) / m
    hist -= mean
    low = hist.min()
    hist *= -mean / low
    hist += mean

    return hist


def postprocess_histogram_mean(hist, p, q, n, m):
    """
    Another strategy. Simply ensures sum(hist) == n. Also not as good.

    hist: np.array() of floats
    """
    extra = float(sum(hist) - n) / m
    hist -= extra
    return hist


def postprocess_histogram_matrix(hist, p, q, n, m):
    """
    generate the perturbation matrix and then find its inverse
    adds more error than the MLE technique

    hist: np.array() of floats
    """
    pmat = np.ones((m, m)) * q
    pmat += np.identity(m) * (p - q)
    ipmat = np.linalg.inv(pmat)
    return np.dot(ipmat, hist)


# this is the one we'll actually use
postprocess_histogram = postprocess_histogram_mle


###############################################################################
##  Perturbation functions  ###################################################
###############################################################################

def perturb_hist_pram(values, m, epsilon, sample,
                      postprocess=postprocess_histogram):
    lam = np.exp(epsilon) / sample  # TODO: is this right?

    # create two blank histograms: one for the real values, one for the
    # perturbed values
    old_hist = np.zeros(m)
    pert_hist = np.zeros(m)

    # the p and q parameters depend on the cardinality of the
    # categorical variable
    p = lam / float(lam + m - 1)
    q = 1 / float(lam + m - 1)

    # random response for each row
    for idx in values:
        # add to the "real" histogram
        old_hist[idx] += 1

        # perturb if necessary
        if random.random() > p - q:
            # pull random index
            idx = random.choice(range(m))

        # sample the whole set
        if random.random() < sample:
            pert_hist[idx] += 1

    # MLE of actual counts
    final_hist = postprocess(pert_hist, p, q, len(values), m)

    return old_hist, final_hist


def perturb_hist_bits(values, m, epsilon, sample,
                      postprocess=postprocess_histogram, rappor=False):
    """
    Perturb each feature subspace separately.
    Each peer sends a bit vector representing the presence or absence of each
    possible feature value.

    If p is provided, this will interpret q from p and epsilon.
    Otherwise, q = 1 - p
    """
    # the ratio p(1-q) / q(1-p)
    # if q = 1-p, this is p**2 / q**2
    lam = np.exp(epsilon)

    if rappor:
        p, q = get_rappor_params(epsilon)
    else:
        p, q = get_privacy_params(m, epsilon)

    # create two blank histograms: one for the real values, one for the
    # perturbed values
    old_hist = np.zeros(m)
    pert_hist = np.zeros(m)

    # random response for each row
    for idx in values:
        # add to the "real" histogram
        old_hist[idx] += 1

        # draw a random set of tuples to return
        myhist = np.random.binomial(1, q, m)

        # draw one random value for the tuple we actually have
        myhist[idx] = np.random.binomial(1, p)
        if sample == 1 or random.random() < sample:
            pert_hist += myhist

    final_hist = postprocess(pert_hist.copy(), p, q, len(values), m)

    return old_hist, final_hist


def perturb_hist_gauss(values, m, epsilon, delta):
    """
    Perturb each feature subspace separately.
    Each peer sends a float vector representing the amount of each possible
    feature value.
    """
    # source: Dwork privacy book, 3.5.3, thm. 3.22
    sigma = 2 * np.log(1.25 / delta) / epsilon

    def hist_elt(idx):
        arr = np.zeros(m)
        arr[idx] += 1
        return arr

    # create two histograms: one for real values, one for perturbed
    old_hist = sum(hist_elt(idx) for idx in values)

    # add some random gaussian noise
    pert_hist = np.random.normal(0, sigma, m) + old_hist

    return old_hist, pert_hist


def generate_partitions(X, n_parts):
    """ generate partitions of a matrix """
    if n_parts > 1:
        folds = KFold(n_splits=n_parts, shuffle=True).split(X)
        return {i: train_ix for i, (_, train_ix) in enumerate(folds)}
    else:
        return {0: np.arange(X.shape[0])}


def dont_perturb(X, y, subsets):
    """
    Generate unperturbed data subsets in the same format as perturb_histograms()
    """
    output = {}
    folds = generate_partitions(X, len(subsets))
    for subset, parts in subsets.items():
        rows = np.concatenate([folds[p] for p in parts])
        indexer = np.ix_(rows, subset)
        output[subset] = X[indexer], y[rows]
    return output


def perturb_histograms(X, y, cardinality, method, epsilon, delta=0, sample=1,
                       perturb_frac=1, perm_eps=None, subsets=None):
    """
    Perturb each feature subspace separately.
    This function takes X and y and converts it into a histogram, then passes it
    on to a histogrm perturbation function.

    X: matrix of real feature data (X[row, column])
    y: array of real labels (y[row])
    method: can be one of 'best', 'pram', 'bits', or 'gauss'
    perm_eps: the epsilon used to permanently perturb the label

    Output: dict mapping each subspace to a perturbed (X, y) pair
    """
    output = {}

    # if epsilon is unspecified or 0 (invalid), don't perturb data
    if method is None or epsilon is None or epsilon == 0:
        return dont_perturb(X, y, subsets)

    # enforce valid perturbation method
    if method not in ['bits', 'rappor', 'pram', 'gauss', 'best']:
        raise ValueError('Method cannot be %r' % method)

    # permanently perturb the y-values
    perm_eps = perm_eps or epsilon
    pert_frac = 1. / (1 + np.exp(perm_eps))
    y_pert = y.copy()

    for i in range(len(y)):
        if random.random() < pert_frac:
            y_pert[i] = not y[i]

    # get the number of possible tuples for a subset
    hsize = lambda subset: int(2 * np.prod([cardinality[s] for s in subset]))

    # convert a tuple to an index into the histogram
    def hist_idx(subset, row):
        res = 0
        base = 1
        for col in subset:
            res += base * X[row][col]
            base *= cardinality[col]
        return res * 2 + y_pert[row]

    # convert the histogram index back into a tuple
    def idx_to_tuple(idx, subset):
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for col in subset:
            my_tup.append(idx % cardinality[col])
            idx /= cardinality[col]
        return my_tup, y

    # array of l2-norm errors
    errs = []

    # partitions of the feature matrix
    n_parts = len(set(sum((p for p in subsets.values()), [])))
    folds = generate_partitions(X, n_parts)

    # iterate over subsets on the outside
    for subset, parts in subsets.items():
        # put together all the partitions this subset will use
        rows = np.concatenate([folds[p] for p in parts])

        m = hsize(subset)
        categoricals = [hist_idx(subset, row) for row in rows]

        # with the 'best' method, figure out what's best for each histogram
        if method == 'best':
            meth = best_perturb_method(m, len(rows), epsilon)
        else:
            meth = method

        if meth == 'pram':
            old_hist, pert_hist = perturb_hist_pram(categoricals, m,
                                                    epsilon, sample)
        elif meth == 'bits':
            old_hist, pert_hist = perturb_hist_bits(categoricals, m,
                                                    epsilon, sample,
                                                    rappor=False)
        elif meth == 'rappor':
            old_hist, pert_hist = perturb_hist_bits(categoricals, m,
                                                    epsilon, sample,
                                                    rappor=True)
        elif meth == 'gauss':
            old_hist, pert_hist = perturb_hist_gauss(categoricals, m,
                                                     epsilon, delta)

        pert_tuples = []    # covariate rows
        labels = []         # label data

        # convert the histogram into a list of rows
        for i, num in enumerate(pert_hist):
            # map histogram index back to tuple
            tup, label = idx_to_tuple(i, subset)

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this
            num = int(round(num))
            pert_tuples += num * [tup]
            labels += num * [label]

        # aand back into a matrix
        out_X = np.array(pert_tuples)
        out_y = np.array(labels)
        output[subset] = out_X, out_y

    return output
