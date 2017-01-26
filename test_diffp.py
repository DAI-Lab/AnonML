import sys
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import factorial, comb
from scipy.stats import binom

ap = argparse.ArgumentParser()
ap.add_argument('--m', type=int, default=2000,
                help='number of possible tuple values')
ap.add_argument('--n', type=int, default=10000, help='number of peers')
ap.add_argument('--p', type=float, default=0.5,
                help='probability each tuple will be perturbed')
ap.add_argument('--plot-real', action='store_true',
                help='plot delta vs real value')
ap.add_argument('--plot-mvn', action='store_true',
                help='plot delta vs the n/m ratio')
ap.add_argument('--plot-m', action='store_true',
                help='plot delta vs m with fixed n/m ratio')

def perturb_prob(m, n, p, real, k):
    """
    Gives the probability that exactly k of a certain row will be sent to the
    aggregator.
        m: total number of possible rows
        n: total number of peers (number of actual rows)
        p: probability that each peer will randomly perturb their row
        real: real number of a certain row present in the dataset
        k: the number of that certain row for which we are trying to assess the probability
    """
    little_p = (1.0 - p) / m
    real_p = p + little_p
    mass = 0

    # probability that i of the real rows will be present
    for i in xrange(min(real, k) + 1):
        # chance that exactly i of the real value holders send this row
        # -times-
        # chance that exactly k - i of the non-real value holders send this row
        mass_i = binom.pmf(i, real, real_p)
        mass_j = binom.pmf(k - i, n - real, little_p)
        mass += mass_i * mass_j

    return mass


def get_delta_range(m, n, p, real):
    # here, we're gonna find delta for a given p and real value
    y1 = {0: perturb_prob(m, n, p, real, 0)}
    y2 = {0: perturb_prob(m, n, p, real + 1, 0)}
    epsilon = get_epsilon(m, p) # actually ln of this but w/e

    for i in xrange(n):
        y1[i] = perturb_prob(m, n, p, real, i)
        y2[i] = perturb_prob(m, n, p, real + 1, i)
        ratio = max(y1[i], y2[i]) / min(y1[i], y2[i])
        if ratio > epsilon and i > 0:
            break

    delta_low = 1.0 - sum(y2.values())
    delta_high = delta_low + y2[i]
    return y1, y2, i, (delta_low, delta_high)


def plot_real_vals(m, n, p, real_vals=None):
    real_vals = real_vals or range(20)
    # here we establish what real value yields the worst delta value
    deltas = []
    indexes = []

    for real in real_vals:
        y1, y2, idx, delta = get_delta_range(m, n, p, real)
        indexes.append(idx)
        deltas.append(delta)

        print real, delta

        # plot probability of each output value given the input value
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        X = sorted(y1.keys())
        #plt.plot(X, y1p)

    #plt.show()

    delta_low = [d[0] for d in deltas]
    delta_high = [d[1] for d in deltas]
    #plt.plot(real_vals, delta_low)
    #plt.plot(real_vals, delta_high)
    #plt.show()
    plt.plot(real_vals, indexes)
    #plt.show()

    return delta_low, delta_high


def plot_m_vs_n(m, p):
    # now we test the effect of n/m on delta (also strictly decreasing)
    fig, ax = plt.subplots(1, 1)
    deltas = []
    all_factors = [i * 0.2 for i in range(5, 50)]

    for f in all_factors:
        n = int(f * m)
        y1, y2, idx, delta = get_delta_range(m, n, p, 0)
        deltas.append(delta)

        print n, m, delta

        # plot probability of each output value given the input value
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        X = sorted(y1.keys())
        ax.plot(X, y1p)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    delta_low = [d[0] for d in deltas]
    delta_high = [d[1] for d in deltas]
    ax.plot(all_factors, delta_low)
    ax.plot(all_factors, delta_high)
    plt.show()


def plot_mn(p):
    # ...and the effect of m, if n remains a constant multiple (logarithmically
    # increasing?)
    fig, ax = plt.subplots(1, 1)
    deltas = []
    all_m = [100, 200, 500, 1000, 5000, 10000, 25000]

    for m in all_m:
        n = m * 5
        y1, y2, delta = get_delta_range(m, n, p, 0)
        deltas.append(delta)

        print n, m, delta

        # plot probability of each output value given the input value
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        X = sorted(y1.keys())
        ax.plot(X, y1p)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    delta_low = [d[0] for d in deltas]
    delta_high = [d[1] for d in deltas]
    ax.plot(all_m, delta_low)
    ax.plot(all_m, delta_high)
    plt.show()


def get_epsilon(m, p):
    # epsilon bound we're going to achieve
    return (1.0 - (1.0 - p) / m) / (1.0 - p - (1.0 - p) / m)


if __name__ == '__main__':
    args = ap.parse_args()

    # probability that a tuple will keep its value after perturbation
    p = 1.0 - args.p
    m = args.m
    n = args.n

    # run our experiments
    if args.plot_real:
        plot_real_vals(m, n, p )
    if args.plot_mvn:
        plot_m_vs_n(m, p)
    if args.plot_m:
        plot_mn(p)


# Note: It seems like, given perturbation factor p, we can achieve
# epsilon-delta differential privacy with an epsilon of ln(1 - p/m) - ln(p - p/m). The delta is a
# function of p, n, and m (n/m?), but this can be made pretty low with some
# good constants.

# e.g.: m = 2k, n = 10k, p = 0.5: Epsilon = ln(2) with delta = 0.004.
