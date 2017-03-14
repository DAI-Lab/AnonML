import io
import os
import pycurl as curl
import stem.process
import requests
from stem.util import term
from rsa_ring_signature import PublicKey, Ring
from Crypto.PublicKey import RSA

SOCKS_PORT = 7000
PROXIES = {
    'http': 'localhost:%d' % SOCKS_PORT,
    'https': 'localhost:%d' % SOCKS_PORT,
}

class Client(object):
    """
    accepts bit string, signs with ring signature, sends message over new tor
    circuit
    """
    def __init__(self, addr, key_size=2048):
        """
        addr: address of the aggregator
        key_size: size of the RSA key pair to generate
        """
        self.agg_addr = addr
        self.my_key = RSA.generate(key_size, os.urandom)
        self.ring = None # needs to be initialized later

    def register(self):
        """
        send a message to the aggergator, acknowledging our intent to take
        part in the learning session and registering our public key
        """
        public_key = PublicKey(self.my_key.e, self.my_key.n, self.my_key.size())
        payload = {'public_key': public_key}
        url = self.agg_addr + '/register'
        requests.post(url, data=payload, proxies=PROXIES)

    def build_ring(self):
        """
        grab the list of public keys from the aggregator and generate a ring
        """
        r = requests.get(self.agg_addr + '/ring')
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
        url = self.agg_addr + '/send_data/' + str(tup_id)
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

    def tor_disconnect(self):
        """ kill the tor process """
        self.tor_ps.kill()


if __name__ == '__main__':
    client = Client('cyphe.rs', 2048)
