import io
import os
import pycurl as curl
import requests
import stem.process
import pdb
import json

from ast import literal_eval as make_tuple
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

class TorClient(object):
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

    def send_data(self, tup_id, bits):
        """
        send a signed message to the aggregator
        """
        data_str = str(tup_id) + str(bits)
        sig = self.ring.sign(self.private_key, self.key_idx, data_str)
        url = self.build_url('send_data')
        payload = {
            'tuple': tup_id,
            'bits': bits,
            'signature': sig,
        }

        try:
            response = requests.post(url, data=payload,  proxies=PROXIES)
        except Exception as e:
            return 'Unable to reach %s (%s)' % (url, e)

        print response

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
    def __init__(self, data_path, subset_path, p_keep, p_change, bin_size):
        """
        data_path: path to featurized data in csv format
        subset_path: path to list of feature subset tuples as string literals
        """
        df = pd.from_csv(data_path)
        self.subsets = []
        with open(subset_path) as f:
            for line in f:
                self.subsets.append(literal_eval(line))

        if subset_path is None:
            # default to one set per variable
            subsets = [(i,) for i in range(self.X.shape[1])]

        assert p_keep > p_change
        self.p_keep = p_keep
        self.p_change = p_change
        self.bin_size = bin_size

        if verbose >= 2:
            print
            if p_change:
                print 'epsilon =', np.log(p_keep / p_change)
            else:
                print 'no perturbation'

        # X: matrix of real feature data (X[row, column])
        # y: array of real labels (y[row])

    def hist_size(self, subset):
        """ get the number of bars a histogram should have """
        return 2 * self.bin_size ** len(subset)

    def hist_index(self, subset, row):
        """ Convert a tuple to an index into the histogram """
        res = 0
        for i, v in enumerate(self.X[row][np.array(subset)]):
            res += self.bin_size ** i * v
        return res * 2 + self.y[row]

    def perturb_and_send(self, verbose=1):
        """
        Perturb each feature subspace and each row separately.
        Send bit vectors representing the presence or absence of each possible
        feature value.
        """
        # iterate over subsets on the outside
        for subset in self.subsets:
            size = self.hist_size(subset)

            # random response for each row
            for row in xrange(self.X.shape[0]):
                # draw a random set of tuples to return
                bits = np.random.binomial(1, p_change, size)

                # calculate the index of our tuple in the list
                idx = hist_index(subset, row)

                # draw one random value for the tuple we actually have
                bits[idx] = np.random.binomial(1, p_keep)
                self.client.send_data(subset, list(bits))
