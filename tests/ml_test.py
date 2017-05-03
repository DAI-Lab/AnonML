#!/usr/bin/env python2.7
import argparse
import pdb
import random
import itertools
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree as sktree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
                             AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring
from anonml.subset_forest import SubsetForest
from perturb import perturb_histogram


TEST_TYPES = ['compare-classifiers', 'subset-size-datasets',
              'perturbation-subset-size', 'perturbation',
              'perturbation-datasets', 'binning-datasets', 'simple']

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
                help='probability of sending any data at all; i.e. sample size')
ap.add_argument('--epsilon', type=float, default=0,
                help="differential privacy parameter. If zero, don't use DP.")
ap.add_argument('--perturb-type', type=str, choices=PERT_TYPES, default='bits',
                help='technique to use to perturb data')
ap.add_argument('--perturb-frac', type=float, default=1,
                help='fraction of users who will do any perturbation at all')

ap.add_argument('--label', type=str, default='dropout',
                help='label we are trying to predict')
ap.add_argument('--subsets', type=str, default=None,
                help='hard-coded subset file')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--recursive-subsets', action='store_true',
                help='generates all subsets that fit inside the largest subset')
ap.add_argument('--num-trials', type=int, default=1,
                help='number of times to try with different, random subsets')
ap.add_argument('--num-folds', type=int, default=3,
                help='number of folds on which to test each classifier')
ap.add_argument('--num-partitions', type=int, default=1,
                help='number of ways to partition the dataset')


###############################################################################
##  Misc helper functions  ####################################################
###############################################################################

def generate_subsets(df, subset_size):
    """
    Generate n_subsets random, non-overlapping subsets of subset_size columns each
    Subsets are lists of column names (strings)
    """
    shuf_cols = list(df.columns)
    random.shuffle(shuf_cols)
    subsets = []

    while len(shuf_cols):
        cols = shuf_cols[:subset_size]
        shuf_cols = shuf_cols[subset_size:]

        subsets.append(cols)
        if args.recursive_subsets:
            for j in range(1, subset_size):
                for c in itertools.combinations(cols, j):
                    subsets.append(c)

    return subsets


###############################################################################
##  Test helper functions  ####################################################
###############################################################################

def test_classifier_unit(clf, training_data, X_test, y_test, metrics):
    """
    Test a classifier on a single test/train fold, and store the results in the
    "results" dict. Can be used in parallel.
    """
    # fit to the training set
    clf.fit(training_data)

    # see what the model predicts for the test set
    y_pred = clf.predict(X_test)
    tp, tn = sum(y_pred & y_test), sum(~y_pred & ~y_test)
    fp, fn = sum(y_pred & ~y_test), sum(~y_pred & y_test)

    if args.verbose >= 2:
        print
        print '\tPredicted/actual true:', sum(y_pred), sum(y_test)
        print '\tPredicted/actual false:', sum(~y_pred), sum(~y_test)
        print '\tTrue positive (rate): %d, %.3f' % (tp, float(tp) /
                                                    sum(y_test))
        print '\tTrue negative (rate): %d, %.3f' % (tn, float(tn) /
                                                    sum(~y_test))
        print '\tFalse positive (rate): %d, %.3f' % (fp, float(fp) /
                                                     sum(~y_test))
        print '\tFalse negative (rate): %d, %.3f' % (fn, float(fn) /
                                                     sum(y_test))

    # score the superclassifier
    results = {}
    for metric in metrics:
        scorer = check_scoring(clf, scoring=metric)
        results[metric] = scorer(clf, X_test, y_test)

    return results


def test_classifier(classifier, df, y, subsets=None, epsilon=None, n_trials=1,
                    n_folds=5, parallel=False, **kwargs):
    """
    Run the given classifier with the given epsilon for n_folds tests, and
    return the results.

    This is where the ** M a G i C ** happens
    """
    X = np.nan_to_num(df.as_matrix())
    y = np.array(y).astype('bool')
    clf = classifier(**kwargs)

    results = {metric: [] for metric in ['roc_auc', 'f1', 'accuracy']}

    if parallel:
        pool = mp.Pool(processes=4)

    for i in range(n_trials):
        # generate n_folds partitions of the data
        folds = KFold(y.shape[0], n_folds=n_folds, shuffle=True)
        if parallel:
            # A list of ApplyResult objects, which will eventually hold the
            # results we want. We'll run through the loop spawning processes,
            # then get all the results at the end.
            fold_results = []

        result = {metric: 0 for metric in ['roc_auc', 'f1', 'accuracy']}
        for train_idx, test_idx in folds:
            # split up data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # find the binning factor of this matrix
            # TODO: should be a parameter, not discovered like this
            bin_size = max([max(X[:,i]) - min(X[:,i]) for i in
                            range(X.shape[1])]) + 1

            # TODO: is this a good way to set delta? (no)
            #       ... is there a good way to set delta?
            delta = np.exp(-float(X_train.shape[0]) / 100)

            # perturb data as a histogram
            training_data = perturb_histogram(X=X_train,
                                              y=y_train,
                                              cardinality=bin_size,
                                              method=args.perturb_type,
                                              epsilon=epsilon,
                                              delta=delta,
                                              subsets=subsets)

            # do parallel thing
            if parallel:
                # watch out this copies a lot of data between processes
                mp_args = (clf, training_data, X_test, y_test, results)
                fold_results.append(pool.apply_async(test_classifier_unit, mp_args))
            else:
                # do serial thing: call the same function but wait for it
                fold_res = test_classifier_unit(clf, training_data, X_test,
                                                y_test, results)
                for met, val in fold_res.items():
                    result[met] += val

        if parallel:
            # collect all the threads
            # there's probably a better way to do this...?
            for r in fold_results:
                fold_res = r.get()
                for met, val in fold_res.items():
                    result[met] += val

            pool.close()
            pool.join()

        for met, val in result.items():
            results[met].append(val / n_folds)

    # just making things numpy arrays so we can do stats easier
    np_f1 = np.array(results['f1'])
    np_auc = np.array(results['roc_auc'])
    np_acc = np.array(results['accuracy'])

    if args.verbose >= 1:
        print
        print 'Results (%s, %d trials):' % (classifier.__name__, n_trials)
        print '\tf1: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_f1.mean(), np_f1.std(), np_f1.min(), np_f1.max())
        print '\tAUC: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_auc.mean(), np_auc.std(), np_auc.min(), np_auc.max())
        print '\tAccuracy: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_acc.mean(), np_acc.std(), np_acc.min(), np_acc.max())

    return clf, {'f1': np_f1, 'auc': np_auc, 'acc': np_acc}


def test_subset_forest(df, labels, epsilon=0, n_trials=1, n_folds=5,
                       subset_size=3, subsets=None, parallel=False):
    subsets = subsets or generate_subsets(df, subset_size)
    subsets_ix = []
    cols = {}

    # subsets: map column names to indices
    for subset in subsets:
        subsets_ix.append(tuple([df.columns.get_loc(c) for c in subset]))
        cols[subsets_ix[-1]] = subset

    results = {met: np.ndarray(0) for met in ['f1', 'auc', 'acc']}
    classifiers = []

    # Test the classifier on each of n_trials different random feature subsets.
    # This function exists because test_classifier does not have the ability to
    # generate a fresh set of subsets between trials.
    for i in range(n_trials):
        clf, res = test_classifier(classifier=SubsetForest, df=df,
                                   y=labels, subsets=subsets_ix, epsilon=epsilon,
                                   n_folds=n_folds, cols=cols)

        # save results in the dictionary, by metric.
        for met, arr in res.items():
            results[met] = np.append(results[met], arr)

        # at the end, we'll sort classifiers by AUC score.
        classifiers.append((res['auc'].mean(), clf))
        subsets = generate_subsets(df, subset_size)

    return sorted(classifiers)[::-1], results


def test_epsilon(df, labels, x, subsets, n_folds):
    """
    Calculate the performance of a classifier for every epsilon in the array x
    """
    y = []
    yerr = []

    for eps in x:
        clf, res = test_classifier(classifier=SubsetForest, df=df,
                                   y=labels, subsets=subsets, epsilon=eps,
                                   n_folds=n_folds)
        mean = res['auc'].mean()
        std = res['auc'].std()
        y.append(mean)
        yerr.append(std)
        if args.verbose >= 1:
            print 'epsilon = %.3f: %.3f (+- %.3f)' % (eps, mean, std)

    return y, yerr


###############################################################################
##  Full tests, with plotting  ################################################
###############################################################################

def compare_classifiers():
    """
    Run a bunch of different classifiers on one dataset and print the results
    """
    df = pd.read_csv(open(args.data_file))
    labels = df[args.label].values
    del df[args.label]

    # load subsets if they're there
    subsets = None
    if args.subsets:
        with open(args.subsets) as f:
            subsets = [[c.strip() for c in l.split(',')] for l in f]

    # save all our data in a dataframe that we can to_csv later.
    columns = []
    for met in ['f1', 'auc', 'acc']:
        columns.append(met + '-mean')
        columns.append(met + '-std')

    classifiers = ['random-forest', 'gradient-boost', 'adaboost',
                   'subset-forest']
    scores = pd.DataFrame(index=classifiers, columns=columns)

    # test our weird whatever
    clfs, res = test_subset_forest(df=df, labels=labels,
                                   epsilon=args.epsilon,
                                   n_trials=args.num_trials,
                                   n_folds=args.num_folds,
                                   subset_size=args.subset_size,
                                   subsets=subsets)
    for met, arr in res.items():
        scores.set_value('subset-forest', met + '-mean', arr.mean())
        scores.set_value('subset-forest', met + '-std', arr.std())

    print
    print 'Top scoring features for best SubsetForest classifier:'
    best_clf = clfs[0][1]
    best_clf.print_scores()

    with open('last-features.txt', 'w') as f:
        for ss, cols in best_clf.cols.iteritems():
            f.write(','.join(cols) + '\n')

    # random forest
    _, res = test_classifier(classifier=RandomForestClassifier, df=df,
                             y=labels, n_trials=args.num_trials,
                             n_folds=args.num_folds, class_weight='balanced')
    for met, arr in res.items():
        scores.set_value('random-forest', met + '-mean', arr.mean())
        scores.set_value('random-forest', met + '-std', arr.std())

    # ADAboost
    _, res = test_classifier(classifier=AdaBoostClassifier, df=df, y=labels,
                             n_trials=args.num_trials, n_folds=args.num_folds)
    for met, arr in res.items():
        scores.set_value('adaboost', met + '-mean', arr.mean())
        scores.set_value('adaboost', met + '-std', arr.std())

    # gradient boosting
    _, res = test_classifier(classifier=GradientBoostingClassifier, df=df,
                             y=labels, n_trials=args.num_trials,
                             n_folds=args.num_folds)
    for met, arr in res.items():
        scores.set_value('gradient-boost', met + '-mean', arr.mean())
        scores.set_value('gradient-boost', met + '-std', arr.std())

    with open(args.out_file, 'w') as f:
        scores.to_csv(f)


def plot_subset_size_datasets():
    """
    Plot performance of a few different datasets across a number of different
    subset sizes
    """
    files = [
        ('baboon_mating/features-b10.csv', 'consort', 'g', 'baboon-mating'),
        ('gender/free-sample-b10.csv', 'class', 'k', 'gender'),
        ('edx/3091x_f12/features-wk10-ld4-b10.csv', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-wk10-ld4-b10.csv', 'dropout', 'b', '6002x'),
    ]
    biggest_subset = 6
    x = range(1, biggest_subset + 1)
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])

    for f, label, fmt, name in files:
        df = pd.read_csv(f)
        labels = df[label].values
        del df[label]

        print
        print 'Testing different subset sizes on dataset', f
        print

        y = []
        yerr = []
        for subset_size in x:
            _, res = test_subset_forest(df=df, labels=labels,
                                        subset_size=subset_size, epsilon=0,
                                        n_trials=args.num_trials,
                                        n_folds=args.num_folds)

            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)
            print '\tsubset size %d: %.3f +- %.3f' % (subset_size, mean, std)

            scores.set_value(subset_size, name + '-mean', mean)
            scores.set_value(subset_size, name + '-std', std)

        if args.plot:
            plt.errorbar(x, y, yerr=yerr, fmt=fmt)

    with open('subset-size-of-datasets.csv', 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0.5, biggest_subset + 0.5, 0.5, 1.0])
        plt.xlabel('subset size')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Subset Size, with Standard Deviation Error')
        plt.show()


def plot_perturbation_subset_size():
    df = pd.read_csv(open(args.data_file))
    labels = df[args.label].values
    del df[args.label]

    biggest_subset = 5
    pairs = zip(range(1, biggest_subset + 1), ['r', 'b', 'g', 'y', 'k'])

    x = [10 ** i/10.0 for i in range(-10, 5)]

    scores = pd.DataFrame(index=x, columns=[str(p[0]) + '-mean' for p in pairs] +
                                           [str(p[0]) + '-std' for p in pairs])

    print
    print 'Testing performance on perturbed data with different feature space sizes'
    print

    n_trials = args.num_trials
    n_folds = args.num_folds

    for subset_size, fmt in pairs:
        print 'Testing perturbation for subset size', subset_size
        results = pd.DataFrame(np.zeros((n_folds * n_trials,
                                         len(x))), columns=x)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for i in range(n_trials):
            print '\ttesting subspace permutation %d/%d, %d folds each' % \
                (i+1, n_trials, n_folds)

            subsets = generate_subsets(df, -1, subset_size)
            subsets_ix = []
            cols = {}
            # subsets: map column names to indices
            for subset in subsets:
                subsets_ix.append(tuple([df.columns.get_loc(c) for c in subset]))
                cols[subsets_ix[-1]] = subset

            for eps in x:
                _, res = test_classifier(classifier=SubsetForest, df=df,
                                         y=labels, subsets=subsets_ix,
                                         epsilon=eps, n_folds=n_folds,
                                         cols=cols)
                start = i * n_folds
                end = (i + 1) * n_folds - 1
                results.ix[start:end, eps] = res['auc']
                print '\t\te = %.2f: %.3f (+- %.3f)' % (eps, res['auc'].mean(),
                                                        res['auc'].std())

        # aggregate the scores for each trial
        for eps in x:
            mean = results[eps].as_matrix().mean()
            std = results[eps].as_matrix().std()
            scores.ix[eps, '%d-mean' % subset_size] = mean
            scores.ix[eps, '%d-std' % subset_size] = std
            print '\te = %.3f: %.3f (+- %.3f)' % (eps, mean, std)

        if args.plot:
            plt.errorbar(x, scores['%d-mean' % subset_size],
                         yerr=scores['%d-std' % subset_size],
                         fmt=fmt)

    with open(args.out_file, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0.0, 1.0, 0.5, 1.0])
        plt.xlabel('perturbation')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def plot_perturbation_datasets():
    """
    Plot performance of a few different datasets by perturbation
    """
    files = [
        ('baboon_mating/features-b5.csv', 'consort', 'g', 'baboon-mating'),
        ('edx/3091x_f12/features-wk10-ld4-b5.csv', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-wk10-ld4-b5.csv', 'dropout', 'b', '6002x'),
        ('gender/free-sample-b5.csv', 'class', 'k', 'gender'),
    ]
    x = [10 ** (i/10.0) for i in range(-10, 5)]
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])

    n_trials = args.num_trials
    n_folds = args.num_folds

    for f, label, fmt, name in files:
        print
        print 'Testing perturbations on dataset', repr(name)
        print
        df = pd.read_csv('data/' + f)
        labels = df[label].values
        del df[label]

        results = pd.DataFrame(np.zeros((n_folds * n_trials,
                                         len(x))), columns=x)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for i in range(n_trials):
            print '\ttesting subspace permutation %d/%d on %s, %d trials each' % \
                (i+1, n_trials, name, n_folds)

            # generate new set of subsets
            subsets = generate_subsets(df, args.subset_size)
            subsets_ix = []
            cols = {}

            # map column names to indices
            for subset in subsets:
                subsets_ix.append(tuple([df.columns.get_loc(c) for c in subset]))
                cols[subsets_ix[-1]] = subset

            for eps in x:
                if args.verbose >= 1:
                    print
                    print 'epsilon =', eps
                _, res = test_classifier(classifier=SubsetForest, df=df,
                                         y=labels, subsets=subsets_ix,
                                         epsilon=eps, n_folds=n_folds,
                                         cols=cols)
                start = i * n_folds
                end = (i + 1) * n_folds - 1
                results.ix[start:end, eps] = res['auc']
                if args.verbose >= 1:
                    print '\te = %.2f: %.3f (+- %.3f)' % (eps, res['auc'].mean(),
                                                          res['auc'].std())


        print '%s results:' % name
        # aggregate the scores for each trial
        for eps in x:
            mean = results[eps].as_matrix().mean()
            std = results[eps].as_matrix().std()
            scores.ix[eps, name+'-mean'] = mean
            scores.ix[eps, name+'-std'] = std
            print '\te = %.3f: %.3f (+- %.3f)' % (eps, mean, std)

        if args.plot:
            plt.errorbar(x, scores[name+'-mean'], yerr=scores[name+'-std'],
                         fmt=fmt)

    with open('pert-by-dataset.csv', 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0.0, 1.0, 0.5, 1.0])
        plt.xlabel('perturbation')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def plot_binning_datasets():
    """
    Plot performance of a few different datasets across a number of bin sizes
    """
    files = [
        ('edx/3091x_f12/features-wk10-ld4%s.csv', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-wk10-ld4%s.csv', 'dropout', 'b', '6002x'),
        ('baboon_mating/features%s.csv', 'consort', 'g', 'baboon-mating'),
        ('gender/free-sample%s.csv', 'class', 'k', 'gender'),
    ]
    x = [0, 20, 10, 5]
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])

    baboon_subsets = []
    with open('baboon_mating/subsets.txt') as f:
        for l in f:
            baboon_subsets.append([c.strip() for c in l.split(',')])

    for f, label, fmt, name in files:
        subsets = None
        if name == 'baboon-mating':
            subsets = baboon_subsets

        print
        print 'Testing different bin sizes on dataset', name
        print

        y = []
        yerr = []
        for bin_size in x:
            extra = '-b%d' % bin_size if bin_size else ''

            # load in the data frame for the binned features
            df = pd.read_csv(f % extra)
            labels = df[label].values
            del df[label]

            print 'loaded dataset', f % extra

            if not subsets:
                subsets = generate_subsets(df, -1, args.subset_size)

            _, res = test_subset_forest(df=df, labels=labels,
                                        n_trials=args.num_trials,
                                        n_folds=args.num_folds,
                                        subset_size=args.subset_size,
                                        subsets=subsets, epsilon=0)
            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)

            scores.set_value(bin_size, name + '-mean', mean)
            scores.set_value(bin_size, name + '-std', std)

        if args.plot:
            plt.errorbar(x, y, yerr=yerr, fmt=fmt)

    with open('performance-by-binsize.csv', 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([-2, 22, 0.5, 1.0])
        plt.xlabel('subset size')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Subset Size, with Standard Deviation Error')
        plt.show()


def simple_test():
    """
    Run one test on the subset forest
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

    # test the silly ensemble
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
        if test == 'compare-classifiers':
            compare_classifiers()
        if test == 'subset-size-datasets':
            plot_subset_size_datasets()
        if test == 'perturbation-subset-size':
            plot_perturbation_subset_size()
        if test == 'perturbation-datasets':
            plot_perturbation_datasets()
        if test == 'binning-datasets':
            plot_binning_datasets()
        if test == 'simple':
            simple_test()


# TODO: plot p = 1-q vs random response vs optimal p, q
#   plot measured error vs performance
#   plot histograms of best decision trees
#   set up experiments with multiple partitions of the dataset

if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
