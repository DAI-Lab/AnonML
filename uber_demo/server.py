import json
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey

app = Flask(__name__)

#accepts bit strings from clients and publishes data, then builds, shares model

@app.route('/send_data/<tup_id>', methods=['POST'])
def post_data(tup_id):
    """ Client sends data and signature """
    data = request.args.get('data')
    signature = request.args.get('signature')

    with open('public_keys.json') as f:
        keys = json.load(f)

    ring = Ring(keys)
    if ring.verify(data, sig):
        data.append(data)
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
    app.run('0.0.0.0', 8000)
