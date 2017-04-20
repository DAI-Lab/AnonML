import numpy as np
import pandas as pd

# accepts bit strings from clients and generates, normalizes histogram

class Aggregator(object):
    def __init__(self, subsets, bin_size, p_keep, p_change):
        """
        subsets: list of tuples of column names
        bin_size: number of bins in each column
        """
        self.bin_size = bin_size
        self.subsets = subsets
        self.p_keep = p_keep
        self.p_change = p_change
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
        return 2 * self.bin_size ** len(subset)

    def index_to_tuple(self, idx, degree):
        """ convert a histogram index back into a tuple """
        my_tup = []
        y = bool(idx % 2)
        idx /= 2
        for _ in range(degree):
            my_tup.append(idx % bin_size)
            idx /= bin_size
        return my_tup, y

    def add_data(self, subset, bits):
        """ add a bit string to the histogram for a particular feature subset """
        self.histograms[tuple(subset)] += np.array(bits).astype(int)

    def renormalize_histogram(self, subset):
        """ renormalize the histogram using the inverse perturbation matrix """
        # generate perturbation matrix
        size = self.hist_size(subset)
        pmat = np.ones((size, size)) * self.p_change
        pmat += np.identity(size) * (self.p_keep - self.p_change)

        # inverse perturbation matrix
        ipmat = np.linalg.inv(pmat)

        # linear algebra
        return np.dot(ipmat, self.histograms[subset])

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
