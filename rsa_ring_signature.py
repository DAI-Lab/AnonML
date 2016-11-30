import os, hashlib, random, Crypto.PublicKey.RSA
import numpy as np

class Ring(object):
    def __init__(self, keys, length=2048):
        self.keys = keys
        self.length = length
        self.n_keys = len(keys)
        self.max_val = 1 << (length - 1)

    def sign(self, message, key_idx):
        self.permut(message)
        sig = [None] * self.n_keys
        u = random.randint(0, self.max_val)
        c = v = self.E(u)

        for i in (range(key_idx + 1, self.n_keys) + range(key_idx)):
            sig[i] = random.randint(0, self.max_val)
            e = self.g(sig[i], self.keys[i].e, self.keys[i].n)
            v = self.E(v^e)
            if (i+1) % self.n_keys == 0:
                c = v

        sig[key_idx] = self.g(v^u, self.keys[key_idx].d, self.keys[key_idx].n)
        return [c] + sig

    def verify(self, message, X):
        self.permut(message)
        def _f(i):
            return self.g(X[i+1], self.keys[i].e, self.keys[i].n)
        y = map(_f, range(len(X)-1))
        def _g(x, i):
            return self.E(x^y[i])
        r = reduce(_g, range(self.n_keys), X[0])
        return r == X[0]

    def permut(self, msg):
        self.p = int(hashlib.sha1('%s' % msg).hexdigest(), 16)

    def E(self, x):
        msg = '%s%s' % (x, self.p)
        return int(hashlib.sha1(msg).hexdigest(), 16)

    def g(self, x, e, n):
        q, r = divmod(x, n)
        if ((q + 1) * n) <= ((1 << self.length) - 1):
            rslt = q * n + pow(r, e, n)
        else:
            rslt = x
        return rslt

if __name__ == '__main__':
    n_keys = 10
    length = 2048
    keys = [Crypto.PublicKey.RSA.generate(length, os.urandom) for i in range(n_keys)]
    ring = Ring(keys, length)
    print ring.sign('hello', 1)
    print ring.sign('hello', 1)
    print ring.sign('world', 1)
    print ring.sign('hello', 2)
    print ring.sign('hello', 3)
