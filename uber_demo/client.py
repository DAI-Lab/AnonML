import io
import os
import pycurl as curl
import requests
import stem.process
import pdb
import json

from stem.util import term
from rsa_ring_signature import PublicKey, Ring
from Crypto.PublicKey import RSA
from flask import Flask

SOCKS_PORT = 7000
PROXIES = {
    'http': 'socks5://127.0.0.1:%d' % SOCKS_PORT,
    'https': 'socks5://127.0.0.1:%d' % SOCKS_PORT,
}

app = Flask(__name__)

class Client(object):
    """
    accepts bit string, signs with ring signature, sends message over new tor
    circuit
    """
    def __init__(self, addr, port=8000, key_size=2048):
        """
        addr: address of the aggregator
        key_size: size of the RSA key pair to generate
        """
        self.agg_addr = addr
        self.agg_port = port
        self.ring = None    # initialized after registration
        self.tor_ps = None  # initialized when tor starts

        print 'generating %d-bit key...' % key_size
        self.my_key = RSA.generate(key_size, os.urandom)
        print 'done'

    def __del__(self):
        self.tor_disconnect()

    def build_url(self, path):
        return 'http://%s:%d/%s' % (self.agg_addr, self.agg_port, path)

    def register(self):
        """
        send a message to the aggergator, acknowledging our intent to take
        part in the learning session and registering our public key
        """
        print 'registering key...'
        public_key = PublicKey(self.my_key.e, self.my_key.n, self.my_key.size())
        payload = public_key.to_json()
        url = self.build_url('register')
        r = requests.post(url, data=payload, proxies=PROXIES)

        if r.status_code == 200:
            print 'success!'
        else:
            print 'error:', r.status_code
            print r.text

    def build_ring(self):
        """
        grab the list of public keys from the aggregator and generate a ring
        """
        r = requests.get(self.agg_addr + 'ring')
        for i, k in enumerate(r.json()['keys']):
            try:
                key = PublicKey(k['e'], k['n'], k['size'])
            except KeyError as e:
                print 'Key at index', i, 'not properly formatted. Missing attribute', e
                return
            all_keys.append(key)

        self.ring = Ring(all_keys)

    def send_data(self, data, tup_id):
        """
        send a signed message to the aggregator
        """
        sig = self.ring.sign(self.private_key, self.key_idx, data)
        url = self.build_url('send_data/' + str(tup_id))
        payload = {
            'data': data,
            'signature': sig,
        }

        try:
            r = requests.post(url, payload,  proxies=PROXIES)
        except Exception as e:
            return 'Unable to reach %s (%s)' % (url, e)

        print r

    def print_bootstrap_lines(self, line):
        if 'Bootastrapped ' in line:
            print term.format(line, term.Color.BLUE)

    def tor_connect(self):
        """ launch the tor process """
        print term.format('Starting Tor:\n', term.Attr.BOLD)

        self.tor_ps = stem.process.launch_tor_with_config(
            config = {'SocksPort': str(SOCKS_PORT)},
            init_msg_handler = self.print_bootstrap_lines,
        )

        # verify that requests works
        print 'tor IP:', requests.get('http://httpbin.org/ip').text

    def tor_disconnect(self):
        """ kill the tor process """
        if self.tor_ps:
            self.tor_ps.kill()

class DataClient(object):
    def perturb_hist_pram(self, X, y, p_keep, p_change, bin_size, subsets=None,
                          verbose=1):
        """
        Perturb each feature subspace separately.
        Each peer sends a bit vector representing the presence or absence of each
        possible feature value.

        X: matrix of real feature data (X[row, column])
        y: array of real labels (y[row])
        Output: dict mapping each subspace to a perturbed (X, y) pair
        """
        assert p_keep > p_change

        if subsets is None:
            # default to one set per variable
            subsets = [(i,) for i in range(X.shape[1])]

        output = {}

        if verbose >= 2:
            print
            if p_change:
                print 'epsilon =', np.log(p_keep / p_change)
            else:
                print 'no perturbation'

        # get the number of possible tuples for a subset
        hsize = lambda subset: 2 * bin_size ** len(subset)

        # function to convert a tuple to an index into the histogram
        def hist_idx(subset, row):
            res = 0
            for i, v in enumerate(X[row][np.array(subset)]):
                res += bin_size ** i * v
            return res * 2 + y[row]

        # function to convert a histogram index back into a tuple
        def idx_to_tuple(idx, degree):
            my_tup = []
            y = bool(idx % 2)
            idx /= 2
            for _ in range(degree):
                my_tup.append(idx % bin_size)
                idx /= bin_size
            return my_tup, y

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

            # calculate L1, L2 norm errors
            l1_err = sum(abs(old_hist - final_hist))
            l2_err = sum((old_hist - final_hist) ** 2)

            #print "Total rows: old = %d, new = %d" % (sum(old_hist), sum(final_hist))
            #print "L1 error = %d, L2 error = %d" % (l1_err, l2_err)

            # aand back into a matrix
            out_X = pd.DataFrame(pert_tuples, columns=subset).as_matrix()
            out_y = np.array(labels)
            output[subset] = out_X, out_y

        return output

if __name__ == '__main__':
    client = Client('cyphe.rs', port=8000, key_size=2048)
    client.tor_connect()
    client.register()
    client.tor_disconnect()
