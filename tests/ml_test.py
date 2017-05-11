#!/usr/bin/env python2.7
import argparse
import pdb
import random
import itertools
import multiprocessing as mp
from collections import defaultdict

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
from sklearn.model_selection import KFold
from sklearn.metrics.scorer import check_scoring
from anonml.subset_forest import SubsetForest
from perturb import perturb_histograms


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
ap.add_argument('--subset-file', type=str, default=None,
                help='hard-coded subsets file')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--recursive-subsets', action='store_true',
                help='generates all subsets that fit inside the largest subset')
ap.add_argument('--num-trials', type=int, default=1,
                help='number of times to try with different, random subsets')

# having this default to 4 is nice for parallelism purposes
ap.add_argument('--num-folds', type=int, default=4,
                help='number of folds on which to test each classifier')
ap.add_argument('--num-partitions', type=int, default=1,
                help='number of ways to partition the dataset')
ap.add_argument('--parallelism', type=int, default=1,
                help='how many processes to run in parallel')


###############################################################################
##  Misc helper functions  ####################################################
###############################################################################

def generate_subspaces(df, subset_size, n_parts=1):
    """
    Generate random, non-overlapping subsets of subset_size columns each
    Subsets are lists of column names (strings)

    subset_size (int): number of columns to include in each subset
    n_parts (int): number of different sets of subsets to generate
    """
    subspaces = {}

    # do k folds of the data, and use each one as a partition
    for i in range(n_parts):
        subspaces[i] = []
        shuf_cols = range(len(df.columns))
        random.shuffle(shuf_cols)

        while len(shuf_cols):
            # pop off the first subset_size columns and add to the list
            subspaces[i].append(tuple(shuf_cols[:subset_size]))
            shuf_cols = shuf_cols[subset_size:]

    return subspaces


def load_subsets(path):
    if path is not None:
        try:
            with open(path) as f:
                return json.load(f)
        except:
            print "Could not open subset file at", path
    return None


###############################################################################
##  Test helper functions  ####################################################
###############################################################################

def test_classifier_once(classifier, kwargs, train_idx, test_idx, metrics,
                         subsets=None, perturb_type=None, epsilon=None):
    """
    Test a classifier on a single test/train fold, and store the results in the
    "results" dict. Can be used in parallel.
    """
    # initialize a new classifier
    clf = classifier(**kwargs)

    # split up data
    X, y = global_X, global_y
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if subsets is None:
        # if we're using the whole dataset, go right ahead
        clf.fit(X_train, y_train)
    else:
        # using subsets:
        # we need to perturb the data first
        # find the binning factor of this matrix
        # TODO: should be a parameter, not discovered like this
        num_bins = max([max(X[:,i]) - min(X[:,i]) for i in
                        range(X.shape[1])]) + 1

        # TODO: is this a good way to set delta? (no)
        #       ... is there a good way to set delta?
        delta = np.exp(-float(X_train.shape[0]) / 100)

        # perturb data as a bunch of histograms
        training_data = perturb_histograms(X=X_train,
                                           y=y_train,
                                           cardinality=num_bins,
                                           method=perturb_type,
                                           epsilon=epsilon,
                                           delta=delta,
                                           subsets=subsets)

        # fit to the training set
        # this function call will only work for SubsetForests
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

    with open('done.txt', 'w') as f:
        f.write('Done!')

    return results


def test_classifier(classifier, df, y, subsets=None, epsilon=None, n_trials=1,
                    n_folds=4, **kwargs):
    """
    Run the given classifier with the given epsilon for n_folds tests, and
    return the results.

    This is where the ** M a G i C ** happens
    """
    X = np.nan_to_num(df.as_matrix())
    y = np.array(y).astype('bool')

    # for each metric, a list of results
    metrics = ['roc_auc', 'f1', 'accuracy']
    results = {metric: [] for metric in metrics}

    # globalize the feature matrix so that the other processes can access it
    global global_X, global_y
    global_X = X
    global_y = y

    if args.verbose >= 1:
        print 'testing %d folds of %d samples' % (n_folds, len(y))

    parallel = args.parallelism > 1
    if parallel:
        pool = mp.Pool(processes=args.parallelism)

    trial_results = []

    # stupid class to enable lazy coding
    # makes normal values act like apply_async results
    class Result(object):
        def __init__(self, value): self.value = value
        def get(self): return self.value

    for i in range(n_trials):
        # generate n_folds subsets of the data
        folds = KFold(n_splits=n_folds, shuffle=True).split(X)

        # A list of ApplyResult objects, which will eventually hold the
        # results we want. We'll run through the loop spawning processes,
        # then get all the results at the end.
        # Only used in parallel
        fold_results = []
        for train_ix, test_ix in folds:
            if parallel:
                if subsets:
                    # send extra args
                    mp_args = (classifier, kwargs, train_ix, test_ix, metrics,
                               subsets, args.perturb_type, epsilon)
                else:
                    # just the basics
                    mp_args = (classifier, kwargs, train_ix, test_ix, metrics)

                # watch out this might copy a lot of data between processes
                apply_result = pool.apply_async(test_classifier_once, mp_args)
                fold_results.append(apply_result)

            else:
                # do serial thing: call the same function but wait for it
                res = test_classifier_once(classifier, kwargs, train_ix,
                                           test_ix, metrics, subsets=subsets,
                                           perturb_type=args.perturb_type,
                                           epsilon=epsilon)
                # store the result in janky wrapper class (~yikes~)
                fold_results.append(Result(res))

        trial_results.append(fold_results)

    # aggregate all the results into big dict thing
    for fold_results in trial_results:
        # a single set of results, metric name -> score
        result = {metric: 0 for metric in metrics}
        for r in fold_results:
            fold_res = r.get()
            for met, val in fold_res.items():
                result[met] += val

        # each value is the sum of n_folds scores, so divide each value by
        # the number of folds and append it to the results list
        for met, val in result.items():
            results[met].append(val / n_folds)

    pool.close()
    pool.join()

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

    return {'f1': np_f1, 'auc': np_auc, 'acc': np_acc}


def test_subset_forest(df, labels, epsilon=0, n_trials=1, n_folds=4,
                       subset_size=3, subsets=None, n_parts=1):
    # map of metric names to arrays of scores (one set for each trial)
    results = {met: np.ndarray(0) for met in ['f1', 'auc', 'acc']}
    classifiers = []

    # Test the classifier on each of n_trials different random feature subsets.
    # This function exists because test_classifier does not have the ability to
    # generate a fresh set of subsets between trials.
    for i in range(n_trials):
        # generate subsets if necessary
        subsets = subsets or generate_subspaces(df, subset_size, n_parts)

        # test the classifier on this set of subsets
        res = test_classifier(classifier=SubsetForest,
                              df=df,
                              y=labels,
                              # the next function takes subsets as idxs
                              subsets=subsets,
                              epsilon=epsilon,
                              n_folds=n_folds,
                              cols=list(df.columns))

        # save results in the dictionary, by metric.
        for met, arr in res.items():
            results[met] = np.append(results[met], arr)

    return results


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

    # load subset data if they're there
    subsets = load_subsets(args.subset_file)

    # save all our data in a dataframe that we can to_csv later.
    columns = []
    for met in ['f1', 'auc', 'acc']:
        columns.append(met + '-mean')
        columns.append(met + '-std')

    classifiers = ['random-forest', 'gradient-boost', 'adaboost',
                   'subset-forest']
    scores = pd.DataFrame(index=classifiers, columns=columns)

    # test our weird whatever
    res = test_subset_forest(df=df, labels=labels,
                             epsilon=args.epsilon,
                             n_trials=args.num_trials,
                             n_folds=args.num_folds,
                             subset_size=args.subset_size,
                             subsets=subsets,
                             n_parts=args.num_partitions)
    for met, arr in res.items():
        scores.set_value('subset-forest', met + '-mean', arr.mean())
        scores.set_value('subset-forest', met + '-std', arr.std())

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
                                        subset_size=subset_size,
                                        epsilon=0,
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

            subsets = generate_subspaces(df, subset_size, args.num_partitions)
            for eps in x:
                res = test_classifier(classifier=SubsetForest,
                                      df=df,
                                      y=labels,
                                      subsets=subsets,
                                      epsilon=eps,
                                      n_folds=n_folds,
                                      cols=list(df.columns))
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
        ('edx/3091x_f12/features-wk10-ld4-b5.csv', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-wk10-ld4-b5.csv', 'dropout', 'b', '6002x'),
        ('baboon_mating/features-b5.csv', 'consort', 'g', 'baboon-mating'),
        ('gender/free-sample-b5.csv', 'class', 'k', 'gender'),
    ]
    x = [np.log(i) for i in np.arange(2, 16)]
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
            subsets = generate_subspaces(df, args.subset_size,
                                         args.num_partitions)
            for eps in x:
                if args.verbose >= 1:
                    print
                    print 'epsilon =', eps

                res = test_classifier(classifier=SubsetForest,
                                      df=df,
                                      y=labels,
                                      subsets=subsets,
                                      epsilon=eps,
                                      n_folds=n_folds,
                                      cols=list(df.columns))
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
        plt.axis([0.0, 2.5, 0.5, 1.0])
        plt.xlabel('epsilon')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Epsilon, with Standard Deviation Error')
        plt.show()


def plot_binning_datasets():
    """
    Plot performance of a few different datasets with different numbers of bins
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

    baboon_subsets = load_subsets('baboon_mating/subsets.txt')

    for f, label, fmt, name in files:
        subsets = None
        if name == 'baboon-mating':
            subsets = baboon_subsets

        print
        print 'Testing different bin sizes on dataset', name
        print

        y = []
        yerr = []
        for num_bins in x:
            extra = '-b%d' % num_bins if num_bins else ''

            # load in the data frame for the binned features
            df = pd.read_csv(f % extra)
            labels = df[label].values
            del df[label]

            print 'loaded dataset', f % extra

            if not subsets:
                subsets = generate_subspaces(df, args.subset_size,
                                             args.num_partitions)

            _, res = test_subset_forest(df=df, labels=labels,
                                        n_trials=args.num_trials,
                                        n_folds=args.num_folds,
                                        subset_size=args.subset_size,
                                        subsets=subsets,
                                        epsilon=0)
            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)

            scores.set_value(num_bins, name + '-mean', mean)
            scores.set_value(num_bins, name + '-std', std)

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

    # load partitions/subsets if they're there
    subsets = load_subsets(args.subset_file)

    # test the silly ensemble
    res = test_subset_forest(df=df, labels=labels,
                             epsilon=args.epsilon,
                             n_trials=args.num_trials,
                             n_folds=args.num_folds,
                             subset_size=args.subset_size,
                             subsets=subsets,
                             n_parts=args.num_partitions)
    print
    for met, arr in res.items():
        print met, 'mean: %.3f, stdev: %.3f' % (arr.mean(), arr.std())


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
