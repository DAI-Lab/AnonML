import json
import requests
import sys
import numpy as np
import pandas as pd
import stem.process

from stem.util import term
from ast import literal_eval as make_tuple
from flask import Flask, jsonify, request
from rsa_ring_signature import Ring, PublicKey
from client import TorClient, DataClient, SOCKS_PORT, PROXIES

tor_ps = None

def print_bootstrap_lines(line):
    if 'Bootastrapped ' in line:
        print term.format(line, term.Color.BLUE)

def tor_connect():
    """ launch the tor process """
    print term.format('Starting Tor:\n', term.Attr.BOLD)

    global tor_ps
    config = {'SocksPort': str(SOCKS_PORT), 'ControlPort': '9051'}
    tor_ps = stem.process.launch_tor_with_config(
        config=config, init_msg_handler=print_bootstrap_lines)

    # verify that requests works
    print 'tor IP:', requests.get('http://httpbin.org/ip', proxies=PROXIES).text

def tor_disconnect():
    """ kill the tor process """
    global tor_ps
    if tor_ps:
        tor_ps.kill()
        tor_ps = None

if __name__ == '__main__':
    # register all peers with the aggregator
    peers = []
    for i in range(3):
        print 'client', i
        tor = TorClient(sys.argv[1], port=8000, key_size=1024)
        tor.register()
        peer = DataClient(tor, data_path='data/demo-data-%d.csv' % i,
                          subset_path='data/subsets.txt', label_col='dropout',
                          bin_size=5, p_keep=0.9, p_change=0.1)
        peers.append(peer)

    print
    print 'registration complete!'
    print

    # have each peer send data anonymously
    tor_connect()
    for p in peers:
        p.perturb_and_send()
    tor_disconnect()
