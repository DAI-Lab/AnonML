import json
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from rsa_ring_signature import Ring

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)


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

    keys.append(request.args.get('key'))

    with open('public_keys.json', 'w') as f:
        json.save(f, keys)


@app.route('/ring', methods=['GET'])
def get_ring():
    """ Return a set of public keys to use in a ring """
    with open('public_keys.json') as f:
        keys = json.load(f)

    return jsonify(keys)


if __name__ == "__main__":
    server = Server(keys)
    app.run()
