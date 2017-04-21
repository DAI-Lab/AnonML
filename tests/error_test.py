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
              'data', 'compare-dist', 'stderr', 'plot-mle']

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
    return np.sqrt(sum(diff_hist ** 2))


def max_likelihood_count(n, p, q, count):
    """ given a noisy bit count, estimate the actual bit count """
    return (est - q * n) / (p - q)


def matrix_se(X, n, p, q):
    """ calculate the standard error of the matrix method """
    pmat = np.ones((X, X)) * q + np.identity(X) * (p - q)
    ipmat = np.linalg.inv(pmat)
    a = ipmat[0, 0]
    b = ipmat[0, 1]

    # standard error of one cell
    se = np.sqrt((a**2 + (X - 1) * b**2) * n * p * q)
    se /= n
    # expected l2 norm of histogram
    return np.sqrt(X) * se


def mle_se(X, n, p, q):
    """ calculate the standard error of the MLE method """
    # standard error of one cell
    base = X * n * q * (1 - q)
    extra = n * (p * (1-p) - q * (1-q))
    total_var = base + extra
    stderr = np.sqrt(total_var) / ((p - q) * n)
    return stderr

    se = np.sqrt(var) / (p - q)
    se /= n
    # expected l2 norm of histogram
    return np.sqrt(X) * se


def plot_mle():
    for eps in EPSILONS:
        P = np.arange(0.01, 1, 0.01)
        y = []
        lam = np.exp(eps)
        for p in P:
            x = p / (lam * (1 - p)) # temp variable for readability
            q = x / (1 + x)
            y.append(mle_se(args.cardinality, args.n_peers, p, q))
        print P[y.index(min(y))], min(y)
        plt.plot(P, y)
    plt.show()


def bitvec_test(epsilons, p):
    values = np.random.randint(0, args.cardinality, size=args.n_peers)
    errs = []
    for eps in epsilons:
        errs.append(l2_error(*perturb_hist_bits(values, args.cardinality, eps,
                                                args.sample, p=p)))
    return errs

def exp_dist(cardinality):
    """
    generate a distribution that looks like an exponential curve
    """
    p = np.exp2(np.arange(cardinality) * -1).astype(float)
    return p / float(sum(p))

def skewed_dist(cardinality):
    """
    generate a distribution that looks like a triangle
    roughly p = x / |X|
    """
    p = np.arange(1, cardinality + 1).astype(float)
    return p / float(sum(p))

def flat_dist(cardinality):
    """
    generate a distribution where all values are equally likely
    """
    return np.ones(cardinality).astype(float) / cardinality

def rand_dist(cardinality):
    """
    generate a distribution with random numbers drawn uniformly from [0, 1).
    """
    p = np.rand(cardinality)
    return p / sum(p)

def test_errors(epsilons, dist=None, method='bits', trials=10):
    """
    dist: numpy array of probabilities for each category
    """
    if method == 'bits':
        pert_func = perturb_hist_bits
    elif method == 'pram':
        pert_func = perturb_hist_pram
    elif method == 'gauss':
        pert_func = perturb_hist_gauss

    errs = []
    values = []

    for t in range(trials):
        values.append(np.random.choice(np.arange(args.cardinality),
                                       size=args.n_peers, replace=True, p=dist))

    for eps in epsilons:
        err = []
        for vals in values:
            # pert_func outputs two histograms, and l2_error accepts two
            # histograms as arguments
            real, pert = pert_func(vals, args.cardinality, eps, args.sample)
            err.append(l2_error(real, pert))
            if err[-1] > 25:
                pdb.set_trace()
        errs.append(err)

    return errs


def compare_distributions():
    ax = plt.subplot()
    ax.set_xscale("log")
    #ax.set_yscale("log")
    eps = EPSILONS
    X = args.cardinality
    N = args.n_peers
    handles = []
    for method in ['bits', 'pram']:
        if method == 'bits':
            trials = 100
        else:
            trials = 10

        errs = equal_synth(eps, method, trials=trials)
        avg_err = [np.mean(err) for err in errs]
        even, = ax.plot(eps, avg_err, label=method+'-uniform')
        for i, err in enumerate(errs):
            ax.plot([eps[i]] * trials, err, 'rx' if method == 'pram' else 'bo')
        #skew, = ax.plot(eps, skewed_synth(eps, method), label=method+'-skew')
        #one, = ax.plot(eps, all_same_synth(eps, method), label=method+'-one')

        handles += [even] #, skew, one]

    errs = []
    for e in eps:
        # bits
        lam = np.exp(e / 2)
        p = lam / float(lam + 1)
        q = 1 / float(lam + 1)
        errs.append(mle_se(X, N, p, q))
    mle_bits, = ax.plot(eps, errs, label='mle-bits')

    errs = []
    for e in eps:
        # bits with fixed p
        lam = np.exp(e)
        p = 0.5
        x = p / (lam * (1 - p)) # temp variable for readability
        q = x / (1 + x)
        errs.append(mle_se(X, N, p, q))
    mle_fixp_bits, = ax.plot(eps, errs, label='mle-fixp-bits')

    errs = []
    for e in eps:
        # pram
        lam = np.exp(e)
        p = lam / float(lam + X - 1)
        q = 1 / float(lam + X - 1)
        errs.append(mle_se(X, N, p, q))
    mle_pram, = ax.plot(eps, errs, label='mle-pram')

    handles += [mle_bits, mle_pram, mle_fixp_bits]

    plt.legend(handles=handles)
    plt.show()


def bitvec_ratio_test():
    ax = plt.subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = EPSILONS
    ps = []
    for p in ps:
        legend, = ax.plot(eps, equal_synth(eps, method),
                          label='p=%.3f, q=%.3f'%(p, q))
    plt.show()


def plot_standard_error():
    ax = plt.subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    eps = EPSILONS
    N = args.n_peers
    X = args.cardinality

    mle_errs = []
    mat_errs = []
    for e in eps:
        # bits
        lam = np.exp(e / 2)
        p = lam / float(lam + 1)
        q = 1 / float(lam + 1)
        mle_errs.append(mle_se(X, N, p, q))
        mat_errs.append(matrix_se(X, N, p, q))

    mle_bits, = ax.plot(eps, mle_errs, label='mle-bits')
    mat_bits, = ax.plot(eps, mat_errs, label='mat-bits')

    mle_errs = []
    mat_errs = []
    for e in eps:
        # pram
        lam = np.exp(e)
        p = lam / float(lam + N - 1)
        q = 1 / float(lam + N - 1)
        mle_errs.append(mle_se(X, N, p, q))
        mat_errs.append(matrix_se(X, N, p, q))

    mle_pram, = ax.plot(eps, mle_errs, label='mle-pram')
    mat_pram, = ax.plot(eps, mat_errs, label='mat-pram')

    plt.legend(handles=[mle_bits, mat_bits, mle_pram, mat_pram])
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
        if test == 'stderr':
            plot_standard_error()
        if test == 'plot-mle':
            plot_mle()

if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
