import pandas as pd
import numpy as np
import ipdb
import random
from anonml.aggregator import get_privacy_params


def postprocess_histogram(hist, p, q, n, m):
    """
    Transform a histogram with negative entries into one with all natural numbers,
    suitable for generating a dataset

    hist: np.array() of floats
    """
    hist -= q * n
    hist /= (p - q)

    return hist


def _postprocess_histogram(hist, p, q, n, m):
    """
    Transform a histogram with negative entries into one with all natural numbers,
    suitable for generating a dataset

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


def _postprocess_histogram(hist, p, q, n, m):
    """
    Transform a histogram with negative entries into one with all natural numbers,
    suitable for generating a dataset

    hist: np.array() of floats
    """
    extra = float(sum(hist) - n) / m
    hist -= extra
    return hist


def _postprocess_histogram(hist, p, q, n, m):
    """
    generate the perturbation matrix and then find its inverse
    adds more error than the MLE technique

    hist: np.array() of floats
    """
    pmat = np.ones((m, m)) * q
    pmat += np.identity(m) * (p - q)
    ipmat = np.linalg.inv(pmat)
    return np.dot(ipmat, hist)


###############################################################################
##  Perturbation functions  ###################################################
###############################################################################

def perturb_hist_pram(values, m, epsilon, sample):
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
    final_hist = postprocess_histogram(pert_hist, p, q, len(values), m)

    return old_hist, final_hist


def perturb_hist_bits(values, m, epsilon, sample, p=None):
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
        if random.random() < sample:
            pert_hist += myhist

    # generate the perturbation matrix and then find its inverse
    ## Not doing it for now because it adds too much error
    #pmat = np.ones((m, m)) * q
    #pmat += np.identity(m) * (p - q)
    #ipmat = np.linalg.inv(pmat)

    final_hist = postprocess_histogram(pert_hist.copy(), p, q, len(values), m)

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


def dont_perturb(X, y, subsets):
    output = {}
    for subset in subsets:
        output[subset] = X[:, np.array(subset)], y.copy()
    return output


def perturb_histogram(X, y, cardinality, method, epsilon, delta=0, sample=1,
                      perturb_frac=1, perm_eps=None, subsets=None):
    """
    Perturb each feature subspace separately.
    This function takes X and y and converts it into a histogram, then passes it
    on to a histogrm perturbation function.

    X: matrix of real feature data (X[row, column])
    y: array of real labels (y[row])
    method: can be one of 'pram', 'bits', or 'gauss'
    perm_eps: the epsilon used to permanently perturb the label

    Output: dict mapping each subspace to a perturbed (X, y) pair
    """
    output = {}

    if epsilon is None or epsilon == 0:
        return dont_perturb(X, y, subsets)

    # permanently perturb the y-values
    perm_eps = perm_eps or epsilon
    pert_frac = 1. / (1 + np.exp(perm_eps))
    y_pert = y.copy()

    for i in range(len(y)):
        if random.random() < pert_frac:
            y_pert[i] = not y[i]

    # get the number of possible tuples for a subset
    hsize = lambda subset: 2 * cardinality ** len(subset)

    # convert a tuple to an index into the histogram
    def hist_idx(subset, row):
        res = 0
        for i, v in enumerate(X[row][np.array(subset)]):
            res += cardinality ** i * v
        return res * 2 + y_pert[row]

    # convert the histogram index back into a tuple
    def idx_to_tuple(idx, degree):
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % cardinality)
            idx /= cardinality
        return my_tup, y

    # array of l2-norm errors
    errs = []

    # iterate over subsets on the outside
    for subset in subsets:
        m = hsize(subset)
        categoricals = [hist_idx(subset, row) for row in xrange(X.shape[0])]

        if method == 'pram':
            # lambda parameter: each peer's real value is lambda times more likely to be
            # reported than any other value.
            old_hist, pert_hist = perturb_hist_pram(categoricals, m, epsilon,
                                                    sample)
        elif method == 'bits':
            # lambda parameter: each peer's real value is lambda times more likely to be
            # reported than any other value.
            old_hist, pert_hist = perturb_hist_bits(categoricals, m, epsilon,
                                                    sample)

        elif method == 'gauss':
            old_hist, pert_hist = perturb_hist_gauss(categoricals, m,
                                                     epsilon, delta)

        pert_tuples = []    # covariate rows
        labels = []         # label data

        # convert the histogram into a list of rows
        for i, num in enumerate(pert_hist):
            # map histogram index back to tuple
            tup, label = idx_to_tuple(i, len(subset))

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this
            num = int(round(num))
            pert_tuples += num * [tup]
            labels += num * [label]

        # aand back into a matrix
        out_X = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
        out_y = np.array(labels)
        output[subset] = out_X, out_y

    return output


def perturb_dataframe(df, epsilon, subsets=None):
    """
    Perturb a whole dataframe at once. Consistency.
    For each row in the dataframe, for each subset of that row, randomly perturb
    all the values of that subset. Subsets must be non-overlapping.

    Input: dataframe of real data
    Output: dataframe of perturbed data
    """
    perturbation = np.exp(epsilon)
    if subsets is None:
        subsets = [[i] for i in df.columns]

    ndf = df.copy()
    index = df.index.to_series()
    for cols in subsets:
        # grab a random sample of indices to perturb
        ix = index.sample(frac=perturbation)

        # perturb the value in each column
        for col in cols:
            ndf.ix[ix, col] = np.random.choice(df[col], m=len(ix))

    return ndf
