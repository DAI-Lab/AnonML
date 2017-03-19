import io
import os
import requests
import pdb
import json
import numpy as np
import pandas as pd

from stem import Signal
from stem.control import Controller
from ast import literal_eval
from rsa_ring_signature import PublicKey, Ring
from Crypto.PublicKey import RSA

SOCKS_PORT = 7000
PROXIES = {
    'http': 'socks5://127.0.0.1:%d' % SOCKS_PORT,
    'https': 'socks5://127.0.0.1:%d' % SOCKS_PORT,
}

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

        print 'generating %d-bit key...' % key_size
        self.my_key = RSA.generate(key_size, os.urandom)
        print 'done'

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

        # no tor proxy here
        r = requests.post(url, data=payload)

        if r.status_code == 200:
            print 'done!'
        else:
            print 'error:', r.status_code
            print r.text
            exit(1)

    def build_ring(self):
        """
        grab the list of public keys from the aggregator and generate a ring
        """
        print 'requesting ring...'
        # no tor proxy here
        r = requests.get(self.build_url('ring'))

        if r.status_code == 200:
            print 'done!'
        else:
            print 'error:', r.status_code
            print r.text
            exit(1)

        all_keys = []
        for i, k in enumerate(r.json()):
            try:
                key = PublicKey(int(k['e']), int(k['n']), int(k['size']))
            except KeyError as e:
                print 'Key at index', i, 'not properly formatted. Missing attribute', e
                return
            all_keys.append(key)

        self.ring = Ring(all_keys)
        print 'initialized ring with %d members' % len(all_keys)

    def key_index(self):
        """ find the index of our key in the ring signature """
        if not self.ring:
            return None
        all_keys = [(k.e, k.n) for k in self.ring.public_keys]
        index = all_keys.index((self.my_key.e, self.my_key.n))
        return index

    def send_data(self, subset, bits):
        """ send a signed message to the aggregator """
        print 'refreshing identity...'
        self.new_identity()

        print 'sending data for subset...'
        data_str = str(subset) + str(bits)
        sig = self.ring.sign(self.my_key, self.key_index(), data_str)
        url = self.build_url('send_data')
        payload = {
            'subset': subset,
            'bits': bits,
            'signature': sig,
        }

        # make sure we are using tor here
        r = requests.post(url, data=payload, proxies=PROXIES)

        if r.status_code == 200:
            print 'done!'
        else:
            print 'error:', r.status_code
            print r.text

    def new_identity(self):
        """ request a new identity from Tor """
        with Controller.from_port(port=9051) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)


class DataClient(object):
    def __init__(self, tor_client, data_path, subset_path, label_col,
                 bin_size=5, p_keep=0.9, p_change=0.1):
        """
        data_path: path to featurized data in csv format
        subset_path: path to list of feature subset tuples as string literals
        """
        self.tor_client = tor_client
        self.df = pd.read_csv(data_path)
        self.label_col = label_col
        self.subsets = []
        with open(subset_path) as f:
            for line in f:
                self.subsets.append(literal_eval(line))

        if subset_path is None:
            # default to one set per variable
            subsets = [(c,) for c in self.df.columns]

        assert p_keep > p_change
        self.p_keep = p_keep
        self.p_change = p_change
        self.bin_size = bin_size

    def hist_size(self, subset):
        """ get the number of bars a histogram should have """
        return 2 * self.bin_size ** len(subset)

    def hist_index(self, subset, row):
        """ Convert a tuple to an index into the histogram """
        res = 0
        for i, v in enumerate(self.df.ix[row, list(subset)]):
            res += self.bin_size ** i * v
        return res * 2 + self.df.ix[row, self.label_col]

    def perturb_and_send(self):
        """
        Perturb each feature subspace and each row separately.
        Send bit vectors representing the presence or absence of each possible
        feature value.
        """
        self.tor_client.build_ring()

        # iterate over subsets on the outside
        for subset in self.subsets:
            size = self.hist_size(subset)

            # random response for each row
            for row in xrange(self.df.shape[0]):
                # draw a random set of tuples to return
                bits = np.random.binomial(1, self.p_change, size)

                # calculate the index of our tuple in the list
                idx = self.hist_index(subset, row)

                # draw one random value for the tuple we actually have
                bits[idx] = np.random.binomial(1, self.p_keep)
                self.tor_client.send_data(subset, list(bits))
