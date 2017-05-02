import numpy as np
import pandas as pd


def get_privacy_params(m, eps):
    """
    given m and epsilon, find the optimal p and q
    """
    lam = np.exp(eps)
    neg_b = lam**2 + m*lam - lam
    rad = np.sqrt((m - 1) * lam**3 + (m**2 - 2*m + 2) * lam**2 + (m - 1) * lam)
    denom = lam**2 - 1

    p1 = (neg_b + rad) / denom
    p2 = (neg_b - rad) / denom

    if p1 >= 0 and p1 <= 1:
        p = p1
    elif p2 >= 0 and p2 <= 1:
        p = p2
    else:
        print 'Error! No p value found. Found', p1, p2

    q = p / (lam * (1 - p) + p)
    return p, q


# accepts bit strings from clients and generates, normalizes histogram

class Aggregator(object):
    def __init__(self, subsets, cardinality, n, p, q):
        """
        subsets: list of tuples of column names
        cardinality: cardinality of each feature
        n: number of peers in the dataset
        p: probability a peer will report a '1' honestly
        q: probability a peer will report a '1' dishonestly
        """
        self.subsets = subsets
        self.cardinality = cardinality
        self.n = n
        self.p = p
        self.q = q

        self.histograms = {}
        self.X = {}     # subset -> features
        self.y = {}     # subset -> labels

        # for each subset, create a new (empty) histogram
        for subset in subsets:
            ss = tuple(subset)
            size = self.hist_size(ss)
            self.histograms[ss] = np.zeros(size)

    def hist_size(self, subset):
        """ get the number of bars a histogram should have """
        return 2 * self.cardinality ** len(subset)

    def index_to_tuple(self, idx, degree):
        """ convert a histogram index back into a tuple """
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % cardinality)
            idx /= cardinality
        return my_tup, y

    def add_data(self, subset, bits):
        """ add a bit string to the histogram for a particular feature subset """
        self.histograms[tuple(subset)] += np.array(bits).astype(int)

    def renormalize_histogram(self, subset):
        """ renormalize the histogram using the maximum likelihood estimate """
        # generate perturbation matrix
        new_hist = self.histograms[subset].copy()
        for i in range(len(new_hist)):
            new_hist[i] = (new_hist[i] - self.q * self.n) / (self.p - self.q)

        self.histograms[subset] = new_hist

    def histogram_to_dataframe(self, subset):
        """
        convert a subset's histogram to a collection of features and labels
        """
        features = []       # feature matrix
        labels = []         # label vector

        # convert the histogram into a list of rows
        for i, num in enumerate(self.histograms[subset]):
            # map histogram index back to tuple
            tup, label = self.index_to_tuple(i, len(subset))

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this?
            num = int(round(num))
            features += num * [tup]
            labels += num * [label]

        # create our feature matrix and label vector
        self.X[subset] = pd.DataFrame(features, columns=subset).as_matrix()
        self.y[subset] = np.array(labels)
