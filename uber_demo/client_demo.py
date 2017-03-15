import json
import numpy as np
import pandas as pd

from ast import literal_eval as make_tuple
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey
from client import TorClient, DataClient

if __name__ == '__main__':
    for i in range(10):
        tor = TorClient('cyphe.rs', port=8000, key_size=1024)
        tor.tor_connect()
        tor.register()
        peer = DataClient(tor, data_path='data/uber_data.csv',
                          subset_path='data/uber_subsets.txt',
                          p_keep=0.9, p_change=0.1, bin_size=5)
        peers.append(peer)



