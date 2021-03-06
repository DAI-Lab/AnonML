import json

from anonml.aggregator import Aggregator
from ast import literal_eval
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey

#accepts bit strings from clients and publishes data, then builds, shares model

PK_PATH = 'data/public_keys.json'

app = Flask(__name__)
agg = None

@app.route('/send_data', methods=['POST'])
def recv_data():
    """ Client sends data and signature """
    data = json.loads(request.form['data'])
    bits = data['bits']
    subset = data['subset']
    signature = data['signature']
    print 'signature:', signature

    keys = []
    with open(PK_PATH) as f:
        for i, k in enumerate(json.load(f)):
            keys.append(PublicKey(int(k['e']), int(k['n']), int(k['size'])))
    ring = Ring(keys)
    data_str = str(subset) + str(bits)
    print 'data str:', data_str

    if ring.verify(data_str, signature):
        agg.add_data(subset, bits)
        return 'success'
    else:
        return 'bad signature', 400


@app.route('/register', methods=['POST'])
def register():
    """ Client says they want to take part in the next round, sends key """
    with open(PK_PATH) as f:
        keys = json.load(f)

    key = {
        'e': request.form['e'],
        'n': request.form['n'],
        'size': request.form['size'],
    }

    if key not in keys:
        print 'registering key', key
        keys.append(key)
    else:
        print 'key already seen!'

    with open(PK_PATH, 'w') as f:
        json.dump(keys, f)

    return 'success'


@app.route('/ring', methods=['GET'])
def get_ring():
    """ Return a set of public keys to use in a ring """
    with open(PK_PATH) as f:
        keys = json.load(f)

    print 'received request for ring; returning %d keys' % len(keys)

    return jsonify(keys)


@app.route('/subsets', methods=['GET'])
def get_subsets():
    """ Return the set of subsets to be used by the clients """
    subsets = []
    with open('data/subsets.txt') as f:
        for line in f:
            subsets.append(literal_eval(line))
    return jsonify(subsets)


if __name__ == "__main__":
    subsets = []
    with open('data/subsets.txt') as f:
        for line in f:
            subsets.append(literal_eval(line))

    with open(PK_PATH, 'w') as f:
        f.write('[]')

    agg = Aggregator(subsets=subsets, bin_size=5, p_keep=0.9, p_change=0.1)
    app.run(host='0.0.0.0', port=8000)
