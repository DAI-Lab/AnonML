import os
import hashlib
import random
import Crypto.PublicKey.RSA

class PublicKey(object):
    def __init__(self, e, n, size):
        self.e = e
        self.n = n
        self.size = size

    def to_json(self):
        return {
            'e': self.e,
            'n': self.n,
            'size': self.size
        }

class Ring(object):
    """
    Object used to create ring signatures.

    Initialize with an (ordered) list of public keys.
    """
    def __init__(self, public_keys):
        self.public_keys = public_keys
        self.length = public_keys[0].size + 1
        self.n_keys = len(public_keys)

        # Make sure all keys are the same size. If they're not, there should be
        # a way to extend them... TODO, I guess
        for k in public_keys:
            assert k.size == self.length - 1

        # maximum value in (0, 2^length - 1)
        # TODO: is this right? should it be (1 << length) - 1?
        self.max_val = 1 << (self.length - 1)

    def sign(self, private_key, key_idx, message):
        """
        Given a message and a private key, generate & return a ring signature
        for the message
        """
        # Step 1. generate a deterministic key for the "encrypt" function
        symkey = self.gen_symkey(message)
        sig = [None] * self.n_keys

        # Step 2. Select an initialization ("glue") value at random in [0, max)
        u = random.randint(0, self.max_val)
        c = v = self.concat_hash(u, symkey)

        # Step 3. Choose a random X[i] for each other ring member that isn't us
        # starting from the next key in the ring, iterate over all of the keys
        # that aren't ours
        for i in (range(key_idx + 1, self.n_keys) + range(key_idx)):

            # choose random value for x[i]
            sig[i] = random.randint(0, self.max_val)

            # compute y for the random x
            e = self.g(sig[i], self.public_keys[i].e, self.public_keys[i].n)

            # update the v and continue along the ring
            v = self.concat_hash(v ^ e, symkey)

            # set c to the v you should have at the end of the ring
            if (i + 1) % self.n_keys == 0:
                c = v

        # Step 4. Solve for y[s], the missing, but now constrained, y value
        sig[key_idx] = self.g(v ^ u, private_key.d, private_key.n)
        return [c] + sig

    def verify(self, message, signature):
        """
        Given a message and a signature (series of X plus v), make sure the
        signature signs the message
        """
        symkey = self.gen_symkey(message)

        # v is the verification value, X is the ring of signatures
        v, X = signature[0], signature[1:]

        # permute an X value to a Y value using the g function
        mapper = lambda i: self.g(X[i], self.public_keys[i].e, self.public_keys[i].n)

        # map the array of x -> array of y
        Y = map(mapper, range(len(X)))

        # XOR the cumulative hash with the next value, then hash that
        reducer = lambda x, i: self.concat_hash(x ^ Y[i], symkey)

        # now do the verification:
        # C(k, v, y[]) = E(k, y[r] ^ E(k, y[r-1] ^ E(... ^ E(k, y[0] ^ v)...)))
        r = reduce(reducer, range(self.n_keys), v)
        return r == v

    def gen_symkey(self, message):
        """
        Compute the symmetric key k as the hash of the message m to be signed
        """
        return int(hashlib.sha1(str(message)).hexdigest(), 16)

    def concat_hash(self, x, symkey):
        """
        Concatenate a number with our symkey and hash it
        This is the E_k function from the paper
        """
        msg = '%s%s' % (x, symkey)
        return int(hashlib.sha1(msg).hexdigest(), 16)

    def g(self, msg, exp, mod):
        """
        Trapdoor rsa function, used for both encyption and decryption.
        Acts as the g() function from the paper.

        mod = n
        exp = e or d
        msg = m
        """
        quotient, remainder = divmod(msg, mod)
        max_val = (1 << self.length) - 1

        if ((quotient + 1) * mod) <= max_val:
            result = quotient * mod + pow(remainder, exp, mod)
        else:
            result = msg

        return result

if __name__ == '__main__':
    n_keys = 10
    length = 1024

    print "generating rsa keys..."
    keys = [Crypto.PublicKey.RSA.generate(length, os.urandom) for i in range(n_keys)]
    public_keys = [PublicKey(k.e, k.n, k.size()) for k in keys]
    print "checking ring..."
    ring = Ring(public_keys)
    sig = ring.sign(keys[1], 1, 'hello')
    ver = ring.verify('hello', sig) and not ring.verify('jello', sig)
    if ver:
        print 'OK!'
    else:
        print 'Not OK.'
