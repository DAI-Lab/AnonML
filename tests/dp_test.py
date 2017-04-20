#!/usr/bin/env python2.7
import sys
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import factorial, comb
from scipy.stats import binom
from scipy.optimize import curve_fit

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
ap.add_argument('--plot-dve', action='store_true',
                help='plot delta vs epsilon with fixed n/m ratio')
ap.add_argument('--plot-evp', action='store_true',
                help='plot epsilon vs delta with fixed m')

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


def get_delta_from_range(m, n, p, real, epsilon=None):
    # here, we're gonna find delta for a given p and real value
    y1 = {0: perturb_prob(m, n, p, real, 0)}
    y2 = {0: perturb_prob(m, n, p, real + 1, 0)}
    epsilon = epsilon or get_epsilon(m, p) # actually ln of this but w/e
    delta = 0

    for i in xrange(n):
        y1[i] = perturb_prob(m, n, p, real, i)
        y2[i] = perturb_prob(m, n, p, real + 1, i)
        bigger = max(y1[i], y2[i])
        smaller = min(y1[i], y2[i])
        ratio = bigger / smaller
        if ratio > epsilon and i > 0:
            delta = max(delta, bigger - smaller * epsilon)
            break

    return y1, y2, delta


def plot_real_vals(m, n, p, real_vals=None):
    real_vals = real_vals or range(20)
    # here we establish what real value yields the worst delta value
    deltas = []

    for real in real_vals:
        y1, y2, delta = get_delta_from_range(m, n, p, real)
        deltas.append(delta)

        print 'real = %d, delta = %.4g' % (real, delta)

        # plot probability of each output value given the input value
        X = sorted(y1.keys())
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        plt.plot(X, y1p)

    plt.show()
    plt.plot(real_vals, deltas)
    plt.show()

    return deltas


def plot_m_vs_n(m, p):
    # now we test the effect of n/m on delta (also strictly decreasing)
    deltas = []
    all_factors = [i * 0.2 for i in range(5, 200)]

    for f in all_factors:
        n = int(f * m)
        delta = 0
        for i in range(10):
            y1, y2, d = get_delta_from_range(m, n, p, real=i)
            if d > delta:
                delta = d
            else:
                break

        deltas.append(delta)

        print 'm = %d, n = %d, real = %d, delta = %.4g' % (m, n, i-1, delta)

        # plot probability of each output value given the input value
        X = sorted(y1.keys())
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        plt.plot(X, y1p)

    plt.show()

    X = np.array(all_factors)
    y = np.log(np.array(deltas))
    popt, pcov = curve_fit(quad, X, y)
    func = lambda x, a, b, c: np.exp(a * x**2 + b * x + c)
    fit_y = func(X, *popt)

    print 'delta = exp(%.3g * (m/n)**2 + %.3g * m/n + %.3g)' % tuple(popt)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(X, deltas)
    #ax.plot(X, fit_y)
    plt.xlabel('N/K')
    plt.ylabel('delta')
    plt.show()


def plot_mn(p, mult=5):
    # ...and the effect of n, if m remains a constant multiple (exponentially increasing)
    deltas = []
    all_m = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    for m in all_m:
        n = m * mult
        delta = 0
        for i in range(5):
            y1, y2, d = get_delta_from_range(m, n, p, real=i)
            if d > delta:
                delta = d
            else:
                break

        deltas.append(delta)

        print 'm = %d, n = %d, delta = %.4g' % (m, n, delta)

        # plot probability of each output value given the input value
        X = sorted(y1.keys())
        y1p = [j[1] for j in sorted(y1.items(), key=lambda k: k[0])]
        plt.plot(X, y1p)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.plot(all_m, deltas)
    plt.show()


def plot_delta_vs_epsilon(p, m=2000, mult=5):
    # plot delta vs. epsilon for fixed m, n, p
    n = m * mult
    deltas = []
    epsilons = [get_epsilon(m, p) * (1 + i * 0.05) for i in range(100)]

    for eps in epsilons:
        delta = 0
        for i in range(5):
            y1, y2, d = get_delta_from_range(m, n, p, real=i, epsilon=eps)
            if d > delta:
                delta = d
            else:
                break

        deltas.append(delta)

        print 'm = %d, n = %d, epsilon = %.3f, delta = %.4g' % (m, n, eps, delta)

    X = np.log(np.array(epsilons))
    y = np.log(np.array(deltas))
    popt, pcov = curve_fit(quad, X, y)
    func = lambda x, a, b, c: np.exp(a * x**2 + b * x + c)
    fit_y = func(X, *popt)

    #print 'y = exp(%.3g * epsilon + %.3g)' % tuple(popt)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(X, deltas)
    #ax.plot(X, fit_y)
    plt.xlabel('epsilon')
    plt.ylabel('delta')
    plt.show()

def plot_epsilon_vs_p(m=2000):
    ps = [i * 0.1 for i in range(1, 10)]
    ps += [0.9 + i * 0.02 for i in range(1, 6)]
    eps = [np.log(get_epsilon(m, 1-p)) for p in ps]
    plt.plot(ps, eps)
    plt.xlabel('p')
    plt.ylabel('epsilon')
    plt.show()

def lin(x, a, b):
    return a * x + b

def quad(x, a, b, c):
    return a * x**2 + b * x + c

def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

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
        plot_real_vals(m, n, p)
    if args.plot_mvn:
        plot_m_vs_n(m, p)
    if args.plot_m:
        plot_mn(p)
    if args.plot_dve:
        plot_delta_vs_epsilon(p)
    if args.plot_evp:
        plot_epsilon_vs_p()


# Note: It seems like, given perturbation factor p, we can achieve
# epsilon-delta differential privacy with an epsilon of ln(1 - p/m) - ln(p - p/m). The delta is a
# function of p, n, and m (n/m?), but this can be made pretty low with some
# good constants.

# e.g.: m = 2k, n = 10k, p = 0.5: Epsilon = ln(2) with delta = 0.004.
