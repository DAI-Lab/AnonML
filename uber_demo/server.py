import json
import numpy as np
import pandas as pd

from ast import literal_eval as make_tuple
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey

#accepts bit strings from clients and publishes data, then builds, shares model

class Aggregator(object):
    def __init__(self, subsets, bin_size, p_keep, p_change):
        """
        subsets: list of tuples
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
            size = self.hist_size(subset)
            self.histograms[subset] = np.zeros(size)

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

    def add_data(self, tup, data):
        """ add a bit string to the histogram for a particular tuple """
        self.histograms[tuple(tup)] += np.array(data)

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
        pert_tuples = []    # feature matrix
        labels = []         # label vector

        # convert the histogram into a list of rows
        for i, num in enumerate(self.histograms[subset]):
            # map histogram index back to tuple
            tup, label = self.index_to_tuple(i, len(subset))

            # round floats to ints, and add that many of the tuple
            # TODO: look into linear programming/other solutions to this?
            num = int(round(num))
            pert_tuples += num * [tup]
            labels += num * [label]

        self.X[subset] = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
        self.y[subset] = np.array(labels)


app = Flask(__name__)
agg = None

@app.route('/send_data', methods=['POST'])
def recv_data():
    """ Client sends data and signature """
    bits = request.args.get('bits')
    tup_id = request.args.get('tuple')
    signature = request.args.get('signature')

    with open('public_keys.json') as f:
        keys = json.load(f)

    ring = Ring(keys)
    data_str = str(tup_id) + str(bits)

    if ring.verify(data_str, sig):
        data.append(data)

        return 'success'
    else:
        return 'bad signature', 400


@app.route('/register', methods=['POST'])
def register():
    """ Client says they want to take part in the next round, sends key """
    with open('public_keys.json') as f:
        keys = json.load(f)

    key = {
        'e': request.form.get('e'),
        'n': request.form.get('n'),
        'size': request.form.get('size'),
    }

    if key not in keys:
        print 'registering key', key
        keys.append(key)
    else:
        print 'key already seen!'

    with open('public_keys.json', 'w') as f:
        json.dump(keys, f)

    return 'success'


@app.route('/ring', methods=['GET'])
def get_ring():
    """ Return a set of public keys to use in a ring """
    with open('public_keys.json') as f:
        keys = json.load(f)

    print 'received request for ring; returning %d keys' % len(keys)

    return jsonify(keys)


if __name__ == "__main__":
    subsets = []
    with open('subsets.txt') as f:
        for line in f:
            subsets.append(literal_eval(line))

    agg = Aggregator(subsets=subsets, bin_size=5, p_keep=0.9, p_change=0.1)
