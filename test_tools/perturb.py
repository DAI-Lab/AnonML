import pandas as pd
import numpy as np
import random

###############################################################################
##  Perturbation functions  ###################################################
###############################################################################

def perturb_hist_pram(X, y, epsilon, bin_size, p_sample=1, subsets=None):
    """
    Perturb each feature subspace separately.
    Each peer sends a bit vector representing the presence or absence of each
    possible feature value.

    X: matrix of real feature data (X[row, column])
    y: array of real labels (y[row])
    Output: dict mapping each subspace to a perturbed (X, y) pair
    """

    if subsets is None:
        # default to one set per variable
        subsets = [(i,) for i in range(X.shape[1])]

    output = {}

    if args.verbose >= 2:
        print
        if p_change:
            print 'epsilon =', np.log(p_keep / p_change)
        else:
            print 'no perturbation'

    # get the number of possible tuples for a subset
    hsize = lambda subset: 2 * bin_size ** len(subset)

    # convert a tuple to an index into the histogram
    def hist_idx(subset, row):
        res = 0
        for i, v in enumerate(X[row][np.array(subset)]):
            res += bin_size ** i * v
        return res * 2 + y[row]

    # convert the histogram index back into a tuple
    def idx_to_tuple(idx, degree):
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % bin_size)
            idx /= bin_size
        return my_tup, y

    # array of l2-norm errors
    errs = []

    # iterate over subsets on the outside
    for subset in subsets:
        size = hsize(subset)
        p_keep = epsilon / float(epsilon + size - 1)
        p_change = 1 / float(epsilon + size - 1)

        print "p_keep:", p_keep, "p_change:", p_change
        print "Hist size:", size

        # create two blank histograms: one for the real values, one for the
        # perturbed values
        old_hist = np.zeros(size)
        pert_hist = np.zeros(size)

        # random response for each row
        for row in xrange(X.shape[0]):
            myhist = np.zeros(size).astype(float)
            # calculate the index of our tuple in the list
            idx = hist_idx(subset, row)

            # add to the "real" histogram
            old_hist[idx] += 1

            # perturb if necessary
            if random.random() > p_keep:
                # pull random index
                idx = random.randint(0, size-1)

            # sample the whole set
            if random.random() < p_sample:
                pert_hist[idx] += 1

        # renormalize the histogram
        pmat = np.ones((size, size)) * p_change
        pmat += np.identity(size) * (p_keep - p_change)
        ipmat = np.linalg.inv(pmat)
        final_hist = np.dot(ipmat, pert_hist)

        pert_tuples = []    # covariate rows
        labels = []         # label data

        # convert the histogram into a list of rows
        for i, num in enumerate(final_hist):
            # map histogram index back to tuple
            tup, label = idx_to_tuple(i, len(subset))

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this
            num = int(round(num))
            pert_tuples += num * [tup]
            labels += num * [label]

        # calculate errors
        diff_hist = old_hist / float(sum(old_hist)) - \
            final_hist / float(sum(final_hist))
        l2_err = sum(diff_hist ** 2)
        errs.append((subset, l2_err))

        # aand back into a matrix
        out_X = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
        out_y = np.array(labels)
        output[subset] = out_X, out_y

    return output, errs

def perturb_hist_bits(X, y, epsilon, bin_size, p_sample=1, subsets=None):
    """
    Perturb each feature subspace separately.
    Each peer sends a bit vector representing the presence or absence of each
    possible feature value.

    X: matrix of real feature data (X[row, column])
    y: array of real labels (y[row])
    Output: dict mapping each subspace to a perturbed (X, y) pair
    """
    p_keep = epsilon / float(epsilon + 1)
    p_change = 1 / float(epsilon + 1)
    assert p_keep > p_change

    if subsets is None:
        # default to one set per variable
        subsets = [(i,) for i in range(X.shape[1])]

    output = {}

    if args.verbose >= 2:
        print
        if p_change:
            print 'epsilon =', np.log(p_keep / p_change)
        else:
            print 'no perturbation'

    # get the number of possible tuples for a subset
    hsize = lambda subset: 2 * bin_size ** len(subset)

    # convert a tuple to an index into the histogram
    def hist_idx(subset, row):
        res = 0
        for i, v in enumerate(X[row][np.array(subset)]):
            res += bin_size ** i * v
        return res * 2 + y[row]

    # convert the histogram index back into a tuple
    def idx_to_tuple(idx, degree):
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % bin_size)
            idx /= bin_size
        return my_tup, y

    # array of l2-norm error
    errs = []

    # iterate over subsets on the outside
    for subset in subsets:
        size = hsize(subset)

        # create two blank histograms: one for the real values, one for the
        # perturbed values
        old_hist = np.zeros(size)
        pert_hist = np.zeros(size)

        # random response for each row
        for row in xrange(X.shape[0]):
            # draw a random set of tuples to return
            myhist = np.random.binomial(1, p_change, size)

            # calculate the index of our tuple in the list
            idx = hist_idx(subset, row)

            # add to the "real" histogram
            old_hist[idx] += 1

            # draw one random value for the tuple we actually have
            myhist[idx] = np.random.binomial(1, p_keep)
            if random.random() < p_sample:
                pert_hist += myhist

        # renormalize the histogram
        pmat = np.ones((size, size)) * p_change
        pmat += np.identity(size) * (p_keep - p_change)
        ipmat = np.linalg.inv(pmat)
        final_hist = np.dot(ipmat, pert_hist)

        pert_tuples = []    # covariate rows
        labels = []         # label data

        # convert the histogram into a list of rows
        for i, num in enumerate(final_hist):
            # map histogram index back to tuple
            tup, label = idx_to_tuple(i, len(subset))

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this
            num = int(round(num))
            pert_tuples += num * [tup]
            labels += num * [label]

        # calculate errors
        diff_hist = old_hist / float(sum(old_hist)) - \
            final_hist / float(sum(final_hist))
        l2_err = sum(diff_hist ** 2)
        errs.append((subset, l2_err))

        # aand back into a matrix
        out_X = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
        out_y = np.array(labels)
        output[subset] = out_X, out_y

    return output, errs


def perturb_hist_gauss(X, y, epsilon, delta, bin_size, subsets=None):
    """
    Perturb each feature subspace separately.
    Each peer sends a float vector representing the amount of each possible
    feature value.

    X: matrix of real feature data (X[row, column])
    y: array of real labels (y[row])
    Output: dict mapping each subspace to a perturbed (X, y) pair
    """
    if subsets is None:
        # default to one set per variable
        subsets = [(i,) for i in range(X.shape[1])]

    output = {}

    print 'epsilon =', epsilon, 'delta =', delta,
    sigma_sq = 2 * np.log(2 / delta) / (epsilon**2)
    print 'R =', sigma_sq

    # get the number of possible tuples for a subset
    hsize = lambda subset: 2 * bin_size ** len(subset)

    # convert a tuple to an index into the histogram
    def hist_idx(subset, row):
        res = 0
        for i, v in enumerate(X[row][np.array(subset)]):
            res += bin_size ** i * v
        return res * 2 + y[row]

    def hist_elt(subset, row):
        arr = np.zeros(hsize(subset))
        arr[hist_idx(subset, row)] += 1
        return arr

    # convert the histogram index back into a tuple
    def idx_to_tuple(idx, degree):
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % bin_size)
            idx /= bin_size
        return my_tup, y

    # array of l2-norm error
    errs = []

    # iterate over subsets on the outside
    for subset in subsets:
        size = hsize(subset)

        # create two blank histograms: one for the real values, one for the
        # perturbed values
        old_hist = sum(hist_elt(subset, row) for row in xrange(X.shape[0]))

        # add some random gaussian noise
        pert_hist = np.random.normal(0, sigma_sq, size) + old_hist

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

        # calculate errors
        diff_hist = old_hist / float(sum(old_hist)) - \
            final_hist / float(sum(final_hist))
        l2_err = sum(diff_hist ** 2)
        errs.append((subset, l2_err))

        # aand back into a matrix
        out_X = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
        out_y = np.array(labels)
        output[subset] = out_X, out_y

    return output, errs


def perturb_histogram(X, y, bin_size, method, epsilon, delta=0, p_sample=1,
                      subsets=None):
    """
    method can be one of 'pram', 'bits', or 'gauss'
    """
    if method == 'bits':
        output, _ = perturb_hist_bits(X, y, bin_size, method, epsilon, p_sample,
                                      subsets)
    if method == 'pram':
        output, _ = perturb_hist_pram(X, y, bin_size, method, epsilon, p_sample,
                                      subsets)
    if method == 'gauss':
        output, _ = perturb_hist_gauss(X, y, bin_size, method, epsilon, delta,
                                       p_sample, subsets)
    return output

def test_errs(subset_size, bin_size, epsilon, delta=0, p_sample=1, subsets=None):
    X = np.random(3, 4)
    for i in range(something):
        _, errs = perturb_hist_bits(X, y, bin_size, method, epsilon, p_sample,
                                    subsets)
        _, errs = perturb_hist_pram(X, y, bin_size, method, epsilon, p_sample,
                                    subsets)
        _, errs = perturb_hist_gauss(X, y, bin_size, method, epsilon, delta,
                                     p_sample, subsets)
    return output


def perturb_dataframe(df, perturbation, subsets=None):
    """
    Perturb a whole dataframe at once. Consistency.
    For each row in the dataframe, for each subset of that row, randomly perturb
    all the values of that subset. Subsets must be non-overlapping.

    Input: dataframe of real data
    Output: dataframe of perturbed data
    """
    if subsets is None:
        subsets = [[i] for i in df.columns]

    ndf = df.copy()
    index = df.index.to_series()
    for cols in subsets:
        # grab a random sample of indices to perturb
        ix = index.sample(frac=perturbation)

        # perturb the value in each column
        for col in cols:
            ndf.ix[ix, col] = np.random.choice(df[col], size=len(ix))

    return ndf

