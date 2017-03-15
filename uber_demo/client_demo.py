import json
import numpy as np
import pandas as pd

from ast import literal_eval as make_tuple
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey
from client import TorClient, DataClient

if __name__ == '__main__':
    tor = TorClient('cyphe.rs', port=8000, key_size=2048)
    tor.tor_connect()
    tor.register()


