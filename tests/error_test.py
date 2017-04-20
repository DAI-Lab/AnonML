#!/usr/bin/env python2.7
import argparse
import pdb
import random
import itertools
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from anonml.subset_forest import SubsetForest
from perturb import *

TEST_TYPES = ['synth-random', 'synth-skewed', 'synth-equal', 'synth-all-same',
              'data', 'compare-dist']

PERT_TYPES = ['bits', 'pram', 'gauss']


ap = argparse.ArgumentParser()
ap.add_argument('tests', type=str, nargs='+', choices=TEST_TYPES,
                help='name of test to run')
ap.add_argument('--data-file', type=str, help='path to the raw data file')
ap.add_argument('--out-file', type=str, help='path to the output csv file')
ap.add_argument('-p', '--plot', action='store_true',
                help='whether to plot the results of the test')
ap.add_argument('-v', '--verbose', type=int, default=0,
                help='how much output to display')

ap.add_argument('--sample', type=float, default=1,
                help='probability that each peer will send any data at all')
ap.add_argument('--epsilon', type=float, default=0,
                help="differential privacy parameter. If zero, don't use DP.")
ap.add_argument('--perturb-type', type=str, choices=PERT_TYPES, default='bits',
                help='technique to use to perturb data')
ap.add_argument('--perturb-frac', type=float, default=1,
                help='fraction of users who will do any perturbation at all')

ap.add_argument('--bin-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--num-trials', type=int, default=1,
                help='number of times to try with different, random subsets')


# options for developing synthetic data
ap.add_argument('--n-peers', type=int, default=10000,
                help='number of peers to generate data for')
ap.add_argument('--cardinality', type=int, default=100,
                help='cardinality of the categorical variable to generate')

EPSILONS = [0.1, 0.25, 0.5, 0.8, 1., 1.5, 2., 3., 4.]


def l2_error(old_hist, pert_hist):
    """ calculate the l2 norm of the type estimation error """
    diff_hist = old_hist / float(sum(old_hist)) - \
        pert_hist / float(sum(pert_hist))
    return sum(diff_hist ** 2)


def bitvec_test(epsilons, p):
    values = np.random.randint(0, args.cardinality, size=args.n_peers)
    errs = []
    for eps in epsilons:
        errs.append(l2_error(*perturb_hist_bits(values, args.cardinality, eps,
                                                args.sample, p=p)))
    return errs

def equal_synth(epsilons, method='bits'):
    if method == 'bits':
        pert_func = perturb_hist_bits
    elif method == 'pram':
        pert_func = perturb_hist_pram
    elif method == 'gauss':
        pert_func = perturb_hist_gauss

    values = np.random.randint(0, args.cardinality, size=args.n_peers)
    errs = []
    for eps in epsilons:
        errs.append(l2_error(*pert_func(values, args.cardinality, eps,
                                        args.sample)))
    return errs


def skewed_synth(epsilons, method='bits'):
    if method == 'bits':
        pert_func = perturb_hist_bits
    elif method == 'pram':
        pert_func = perturb_hist_pram
    elif method == 'gauss':
        pert_func = perturb_hist_gauss

    a = np.arange(args.cardinality).astype(float)
    p = np.arange(1, args.cardinality + 1).astype(float)
    p /= float(sum(p))
    values = np.random.choice(a, size=args.n_peers, replace=True, p=p)
    errs = []
    for eps in epsilons:
        errs.append(l2_error(*pert_func(values, args.cardinality, eps,
                                        args.sample)))
    return errs


def all_same_synth(epsilons, method='bits'):
    if method == 'bits':
        pert_func = perturb_hist_bits
    elif method == 'pram':
        pert_func = perturb_hist_pram
    elif method == 'gauss':
        pert_func = perturb_hist_gauss

    values = np.array([0] * args.n_peers)
    errs = []
    for eps in epsilons:
        errs.append(l2_error(*pert_func(values, args.cardinality, eps,
                                        args.sample)))
    return errs


def compare_distributions():
    ax = plt.subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = EPSILONS
    handles = []
    for method in ['bits', 'pram']:
        even, = ax.plot(eps, equal_synth(eps, method), label=method + '-even')
        skew, = ax.plot(eps, skewed_synth(eps, method), label=method + '-skew')
        one, = ax.plot(eps, all_same_synth(eps, method), label=method + '-one')
        handles += [even, skew, one]

    plt.legend(handles=handles)
    plt.show()


def test_bitvec_ratio():
    ax = plt.subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = EPSILONS
    for TODO in TODO:
        even, = ax.plot(eps, equal_synth(eps, method), label=method + '-even')
    plt.show()


def data_test():
    """
    required args: data-file, epsilon
    """
    df = pd.read_csv(open(args.data_file))
    labels = df[args.label].values
    del df[args.label]

    # load subsets if they're there
    subsets = None
    if args.subsets:
        with open(args.subsets) as f:
            subsets = [[c.strip() for c in l.split(',')] for l in f]

    # test the silly ensemple
    clfs, res = test_subset_forest(df=df, labels=labels,
                                   epsilon=args.epsilon,
                                   n_trials=args.num_trials,
                                   n_folds=args.num_folds,
                                   subset_size=args.subset_size,
                                   subsets=subsets)
    print
    for met, arr in res.items():
        print met, 'mean: %.3f, stdev: %.3f' % (arr.mean(), arr.std())

    print
    print 'Top scoring features for best SubsetForest classifier:'
    best_clf = clfs[0][1]
    best_clf.print_scores()

def main():
    for test in args.tests:
        if test == 'compare-dist':
            compare_distributions()

if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
