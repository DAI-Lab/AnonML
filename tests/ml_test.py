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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.scorer import check_scoring
from sklearn.linear_model import LinearRegression, Lasso
from anonml.subset_forest import SubsetForest
from perturb import perturb_histograms
from make_features import bucket_data


TEST_TYPES = ['simple', 'compare-classifiers',
              'binning-datasets', 'subset-size-datasets',
              'perturbation', 'perturbation-subset-size',
              'perturbation-datasets', 'perturbation-partitions',
              'perturbation-method', 'method-subset-size']

PERT_TYPES = ['bits', 'rappor', 'pram', 'gauss', 'best']

METRICS = ['f1', 'roc_auc', 'accuracy']

ap = argparse.ArgumentParser()
ap.add_argument('tests', type=str, nargs='+', choices=TEST_TYPES,
                help='name of test to run')
ap.add_argument('--data-file', type=str, help='path to the raw data file')
ap.add_argument('--out-file', type=str, help='path to the output csv file')
ap.add_argument('-p', '--plot', action='store_true',
                help='whether to plot the results of the test')
ap.add_argument('--plot-roc', action='store_true',
                help='whether to plot the ROC curve after every trial run')
ap.add_argument('-v', '--verbose', type=int, default=0,
                help='how much output to display')

ap.add_argument('--sample', type=float, default=1,
                help='probability of sending any data at all; i.e. sample size')
ap.add_argument('--epsilon', type=float, default=0,
                help="differential privacy parameter. If zero, don't use DP.")
ap.add_argument('--budget', type=float, default=0,
                help='total privacy budget. Defaults to (eps * len(subsets)).')
ap.add_argument('--perturb-type', type=str, choices=PERT_TYPES, default='best',
                help='technique to use to perturb data')
ap.add_argument('--perturb-frac', type=float, default=1,
                help='fraction of users who will do any perturbation at all')

ap.add_argument('--label', type=str, default='dropout',
                help='label we are trying to predict')
ap.add_argument('--feature-file', type=str, default=None,
                help='if provided, only use features named in this file'
                ' (new-line separated)')
ap.add_argument('--subset-file', type=str, default=None,
                help='hard-coded subsets file')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--clf-metric', type=str, default='f1', choices=METRICS,
                help='metric by which to weight subclassifiers in the subset forest')

# having this default to 4 is nice for parallelism purposes
ap.add_argument('--num-folds', type=int, default=4,
                help='number of folds on which to test each classifier')
ap.add_argument('--num-trials', type=int, default=1,
                help='number of times to try with different, random subsets')
ap.add_argument('--trials-per-subset', type=int, default=1,
                help='number of times to try with each subset')
ap.add_argument('--num-partitions', type=int, default=1,
                help='number of ways to partition the dataset')
ap.add_argument('--num-subsets', type=int, default=0,
                help='number of subsets to collect from each peer.'
                ' Defaults to budget / epsilon.')
ap.add_argument('--parallelism', type=int, default=1,
                help='how many processes to run in parallel')



###############################################################################
##  Misc helper functions  ####################################################
###############################################################################

def generate_subspaces(n_cols, subset_size, n_parts=1, n_subsets=None):
    """
    Generate random, non-overlapping subsets of subset_size columns each
    Subsets are lists of column names (strings)

    subset_size (int): number of columns to include in each subset
    n_parts (int): number of partitions to generate
    """
    subspaces = defaultdict(list)

    # do k folds of the data, and use each one as a partition
    for i in range(n_parts):
        shuf_cols = range(n_cols)
        random.shuffle(shuf_cols)

        # if necessary, truncate to a random subset of the data
        # n_subsets cannot be 0
        if n_subsets:
            shuf_cols = shuf_cols[:subset_size * n_subsets]

        while len(shuf_cols):
            # pop off the first subset_size columns: that's the new subset
            # sort its elements, since order doesn't matter
            subset = tuple(sorted(shuf_cols[:subset_size]))

            # add this partition to this subset's list
            subspaces[subset].append(i)

            # truncate the remaining columns
            shuf_cols = shuf_cols[subset_size:]

    return dict(subspaces)


def load_subsets(path):
    if path is not None:
        try:
            with open(path) as f:
                return json.load(f)
        except:
            print "Could not open subset file at", path
    return None


# stupid class to enable lazy coding
# makes normal values act like apply_async results
class SerialResult(object):
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def load_csv(path, feature_file=None):
    """
    load set of features that we're allowed to use - these must be mutually
    independent.
    """
    df = pd.read_csv(open(path))
    if feature_file is not None:
        try:
            feats = []
            with open(feature_file) as ff:
                for line in ff:
                    args = line.strip().split()
                    feat = args[0]
                    feats.append(feat)

                    # if there is an optional type argument, cast with that
                    if len(args) > 1:
                        df[feat] = df[feat].astype(args[1])

            df = df[feats]
        except Exception as e:
            print e
            print "Could not open feature file at", feature_file
    return df


###############################################################################
##  Test helper functions  ####################################################
###############################################################################

global_total = defaultdict(float)

def test_classifier_once(classifier, kwargs, train_idx, test_idx, metrics,
                         subsets=None, perturb_type=None, epsilon=None,
                         bucket=False):
    """
    Test a classifier on a single test/train fold, and store the results in the
    "results" dict. Can be used in parallel.

    bucket: whether to do private bucketing of features at the last minute
    """
    if args.verbose >= 2:
        print "Spawning test thread!"

    # initialize a new classifier
    clf = classifier(**kwargs)

    # gather global data
    df, y = global_df, global_y

    # bucket data if necessary
    if bucket:
        df = bucket_data(df, 2, label=None, privacy=epsilon,
                         verbose=args.verbose)

    # get data into the right format
    X = np.nan_to_num(df.as_matrix())
    y = np.array(y).astype('bool')

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if classifier != SubsetForest:
        # if we're using a normal classifier, go right ahead
        clf.fit(X_train, y_train)
    else:
        # using subsets:
        # we need to perturb the data first
        # find the binning factor of this matrix
        # TODO: should be a parameter, not discovered like this
        cardinality = {i: max(X[:,i]) - min(X[:,i]) + 1 for i in range(X.shape[1])}

        # TODO: is this a good way to set delta? (no)
        #       ... is there a good way to set delta?
        delta = np.exp(-float(X_train.shape[0]) / 100)

        if args.verbose >= 2:
            print 'Perturbing histogram:'
            print 'cardinality', cardinality

        # perturb data as a bunch of histograms
        training_data = perturb_histograms(X=X_train,
                                           y=y_train,
                                           cardinality=cardinality,
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

        tree_scores = []
        for subset, tree in clf.classifiers.items():
            scorer = check_scoring(tree, scoring='roc_auc')
            tree_scores.append((-scorer(tree, X_test[:, subset], y_test),
                                clf.scores[subset]['roc_auc'],
                                [clf.cols[s] for s in subset]))
            for s in subset:
                global_total[clf.cols[s]] += -tree_scores[-1][0]

        for ts in sorted(tree_scores)[:3]:
            print '\tpredicted: %.3f, actual: %.3f' % (ts[1], -ts[0]), ts[2]

    if args.plot_roc:
        y_scores = clf.predict_proba(X_test)
        roc_pert = roc_curve(y_test, y_scores[:,1])
        auc_pert = auc(roc_pert[0], roc_pert[1])

        # compare against unperturbed data
        training_data_clean = perturb_histograms(X=X_train, y=y_train,
                                                 cardinality={},
                                                 epsilon=None,
                                                 method=None,
                                                 subsets=subsets)
        clf.fit(training_data_clean)
        y_scores = clf.predict_proba(X_test)
        roc_clean = roc_curve(y_test, y_scores[:,1])
        auc_clean = auc(roc_clean[0], roc_clean[1])

        # plot them both
        plt.plot(roc_pert[0], roc_pert[1], color='darkorange',
                 label='Perturbed data ROC curve (area = %0.2f)' % auc_pert)
        plt.plot(roc_clean[0], roc_clean[1], color='darkgreen',
                 label='Clean data ROC curve (area = %0.2f)' % auc_clean)

        # baseline line
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.show()

    # score the superclassifier
    results = {}
    for metric in metrics:
        scorer = check_scoring(clf, scoring=metric)
        try:
            results[metric] = scorer(clf, X_test, y_test)
        except:
            pdb.set_trace()

    return results


def test_classifier(classifier, df, y, subsets=None, subset_size=None,
                    n_parts=1, n_subsets=None, epsilon=None, perturb_type=None,
                    n_trials=1, n_folds=4, bucket=False, **kwargs):
    """
    Run the given classifier with the given epsilon for n_folds tests, and
    return the results.

    This is where the ** M a G i C ** happens
    """
    # for each metric, a list of results
    metrics = ['roc_auc', 'f1', 'accuracy']
    results = {metric: [] for metric in metrics}

    # globalize the feature matrix so that the other processes can access it
    global global_df, global_y
    global_df = df
    global_y = y

    # if we're passed subsets, use those every time; otherwise generate them
    perm_subsets = subsets

    if args.verbose >= 1:
        print 'testing %d trials on %d samples, %d folds each' % (
            n_trials, len(y), n_folds)

    # optional parallelism
    parallel = args.parallelism > 1
    if parallel:
        pool = mp.Pool(processes=args.parallelism)

    # list of fold results
    trial_results = []

    for i in range(n_trials):
        # generate n_folds slices of the data
        folds = KFold(n_splits=n_folds, shuffle=True).split(df.as_matrix())

        # generate subsets if necessary
        if classifier == SubsetForest:
            subsets = perm_subsets or generate_subspaces(df.shape[1], subset_size,
                                                         n_parts, n_subsets)

        # A list of ApplyResult objects, which will eventually hold the
        # results we want. We'll run through the loop spawning processes,
        # then get all the results at the end.
        # Only used in parallel
        fold_results = []
        for train_ix, test_ix in folds:
            if parallel:
                mp_args = [classifier, kwargs, train_ix, test_ix, metrics]
                if classifier == SubsetForest:
                    # send extra args
                    mp_args += [subsets, perturb_type, epsilon, bucket]

                # watch out this might copy a lot of data between processes
                apply_result = pool.apply_async(test_classifier_once,
                                                tuple(mp_args))
                fold_results.append(apply_result)

            else:
                # do serial thing: call the same function but wait for it
                res = test_classifier_once(classifier, kwargs, train_ix,
                                           test_ix, metrics, subsets=subsets,
                                           perturb_type=perturb_type,
                                           epsilon=epsilon, bucket=bucket)

                # store the result in janky wrapper class (~yikes~)
                fold_results.append(SerialResult(res))

        trial_results.append(fold_results)

    if parallel:
        # tell the pool to close worker processes once it's done (.close()), then
        # wait for everything to finish and exit (.join())
        pool.close()
        pool.join()

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

    # just making things numpy arrays so we can do stats easier
    np_f1 = np.array(results['f1'])
    np_auc = np.array(results['roc_auc'])
    np_acc = np.array(results['accuracy'])

    if args.verbose >= 1:
        print
        print 'Results (%s, %d trials):' % (classifier.__name__, n_trials)
        print '\tf1: mean = %f, stdev = %f, (%.3f, %.3f)' % \
            (np_f1.mean(), np_f1.std(), np_f1.min(), np_f1.max())
        print '\tAUC: mean = %f, stdev = %f, (%.3f, %.3f)' % \
            (np_auc.mean(), np_auc.std(), np_auc.min(), np_auc.max())
        print '\tAccuracy: mean = %f, stdev = %f, (%.3f, %.3f)' % \
            (np_acc.mean(), np_acc.std(), np_acc.min(), np_acc.max())

    return {'f1': np_f1, 'auc': np_auc, 'acc': np_acc}


###############################################################################
##  Full tests, with plotting  ################################################
###############################################################################

## Part 1: no differential privacy

def compare_classifiers():
    """
    Run a bunch of different classifiers on one dataset and print the results
    """
    df = load_csv(args.data_file, args.feature_file)
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
    res = test_classifier(classifier=SubsetForest, df=df, y=labels,
                          subsets=subsets,
                          subset_size=args.subset_size,
                          n_parts=args.num_partitions,
                          n_subsets=args.num_subsets,
                          epsilon=args.epsilon,
                          perturb_type=args.perturb_type,
                          n_trials=args.num_trials,
                          n_folds=args.num_folds)
    for met, arr in res.items():
        scores.set_value('subset-forest', met + '-mean', arr.mean())
        scores.set_value('subset-forest', met + '-std', arr.std())

    # random forest
    res = test_classifier(classifier=RandomForestClassifier, df=df,
                          y=labels, n_trials=args.num_trials,
                          n_folds=args.num_folds, class_weight='balanced')
    for met, arr in res.items():
        scores.set_value('random-forest', met + '-mean', arr.mean())
        scores.set_value('random-forest', met + '-std', arr.std())

    # adaboost
    res = test_classifier(classifier=AdaBoostClassifier, df=df, y=labels,
                          n_trials=args.num_trials, n_folds=args.num_folds)
    for met, arr in res.items():
        scores.set_value('adaboost', met + '-mean', arr.mean())
        scores.set_value('adaboost', met + '-std', arr.std())

    # gradient boosting
    #res = test_classifier(classifier=GradientBoostingClassifier, df=df,
                          #y=labels, n_trials=args.num_trials,
                          #n_folds=args.num_folds)
    #for met, arr in res.items():
        #scores.set_value('gradient-boost', met + '-mean', arr.mean())
        #scores.set_value('gradient-boost', met + '-std', arr.std())

    outfile = args.out_file or 'compare-classifiers.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)


def plot_binning_datasets():
    """
    Plot performance of a few different datasets with different numbers of bins
    """
    files = [
        ('edx/3091x_f12/features-all-wk10-ld4.csv', 'edx-feats.txt', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-all-wk10-ld4.csv', 'edx-feats.txt', 'dropout', 'b', '6002x'),
        ('census/features.csv', 'census-feats.txt', 'label', 'g', 'census'),
    ]
    x = [0, 10, 5, 2]
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])
    scores.index.name = 'perturbation'

    for f, feats, label, fmt, name in files:
        subsets = None

        print
        print 'Testing different bin sizes on dataset', name
        print

        y = []
        yerr = []
        df = load_csv('data/' + f, feats)
        labels = df[label].values
        del df[label]

        for num_bins in x:
            extra = '-b%d' % num_bins if num_bins else ''

            # generate bucketed data from continuous features
            df = bucket_data(df, num_bins, label=label, private=False,
                             verbose=args.verbose)

            print 'generated %d bins for dataset %s' % (num_bins, name)

            if not subsets:
                subsets = generate_subspaces(df.shape[1], args.subset_size,
                                             args.num_partitions,
                                             args.num_subsets)

            res = test_classifier(classifier=SubsetForest,
                                  df=df, y=labels,
                                  subsets=subsets,
                                  subset_size=args.subset_size,
                                  n_parts=args.num_partitions,
                                  n_subsets=args.num_subsets,
                                  epsilon=0,
                                  perturb_type=args.perturb_type,
                                  n_trials=args.num_trials,
                                  n_folds=args.num_folds,
                                  bucket=False)

            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)

            scores.set_value(num_bins, name + '-mean', mean)
            scores.set_value(num_bins, name + '-std', std)

        if args.plot:
            plt.errorbar(x, y, yerr=yerr, fmt=fmt)

    outfile = args.out_file or 'auc-by-binsize.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([-2, 22, 0.5, 1.0])
        plt.xlabel('subset size')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Subset Size, with Standard Deviation Error')
        plt.show()


def plot_subset_size_datasets():
    """
    Plot performance of a few different datasets across a number of different
    subset sizes
    """
    files = [
        ('edx/3091x_f12/features-all-wk10-ld4.csv', 'edx-feats.txt', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-all-wk10-ld4.csv', 'edx-feats.txt', 'dropout', 'b', '6002x'),
        ('census/features.csv', 'census-feats.txt', 'label', 'g', 'census'),
    ]
    biggest_subset = 5
    x = range(1, biggest_subset + 1)
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])

    for f, feats, label, fmt, name in files:
        df = pd.read_csv(f)
        labels = df[label].values
        del df[label]

        print
        print 'Testing different subset sizes on dataset', f
        print

        y = []
        yerr = []
        for subset_size in x:
            res = test_classifier(classifier=SubsetForest, df=df, y=labels,
                                  subsets=subsets,
                                  subset_size=subset_size,
                                  n_subsets=args.num_subsets,
                                  n_parts=args.num_partitions,
                                  epsilon=0,
                                  n_trials=args.num_trials,
                                  n_folds=args.num_folds,
                                  bucket=True)

            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)
            print '\tsubset size %d: %.3f +- %.3f' % (subset_size, mean, std)

            scores.set_value(subset_size, name + '-mean', mean)
            scores.set_value(subset_size, name + '-std', std)

        if args.plot:
            plt.errorbar(x, y, yerr=yerr, fmt=fmt)

    outfile = args.out_file or 'subset-size-of-datasets.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0.5, biggest_subset + 0.5, 0.5, 1.0])
        plt.xlabel('subset size')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Subset Size, with Standard Deviation Error')
        plt.show()


################################################################################
## Part 2: differential privacy
################################################################################

def plot_perturbation_subset_size():
    """
    Plot performance as a function of subset size, for fixed privacy budget
    """
    df = load_csv(args.data_file, args.feature_file)
    labels = df[args.label].values
    del df[args.label]

    biggest_subset = 5
    sizes = range(1, biggest_subset + 1)
    budget = np.linspace(0.5, 5, 15)

    scores = pd.DataFrame(index=budget, columns=[str(s) + '-mean' for s in sizes] +
                                                [str(s) + '-std' for s in sizes])

    print
    print 'Testing performance on perturbed data with different feature subset sizes'
    print

    n_trials = args.num_trials
    n_folds = args.num_folds
    n_parts = args.num_partitions
    n_subsets = args.num_subsets

    for subset_size in sizes:
        print 'Testing perturbation for subset size', subset_size
        shape = (n_trials, len(budget))
        results = pd.DataFrame(np.zeros(shape), columns=budget)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for b in budget:
            eps = b / float(n_subsets)
            res = test_classifier(classifier=SubsetForest,
                                  df=df,
                                  y=labels,
                                  epsilon=eps,
                                  perturb_type=args.perturb_type,
                                  n_trials=n_trials,
                                  n_folds=n_folds,
                                  n_parts=n_parts,
                                  n_subsets=n_subsets,
                                  subset_size=subset_size,
                                  bucket=True,
                                  cols=list(df.columns))

            results[b] = res['auc']
            err = res['auc'].std() / np.sqrt(n_trials)
            print
            print 'subset size %d, budget = %.2f: %.3f (+- %.3f)' % (
                subset_size, b, res['auc'].mean(), err)
            print

        # aggregate the scores for each trial
        for b in budget:
            mean = results[b].as_matrix().mean()
            std = results[b].as_matrix().std()
            scores.ix[b, '%d-mean' % subset_size] = mean
            scores.ix[b, '%d-std' % subset_size] = std
            print '\tbudget = %.3f: %.3f (+- %.3f)' % (b, mean, std)

        if args.plot:
            plt.errorbar(budget, scores['%d-mean' % subset_size],
                         yerr=scores['%d-std' % subset_size])

    outfile = args.out_file or 'perturbation-subset-size.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0, 5.0, 0.5, 1.0])
        plt.xlabel('epsilon')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def plot_perturbation_partitions():
    """
    Plot performance as a function of number of partitions, for fixed budget.
    """
    df = load_csv(args.data_file, args.feature_file)
    labels = df[args.label].values
    del df[args.label]

    partitions = [1, 8, 64] #[2**i for i in range(7)]
    budget = np.linspace(0.5, 5, 15)
    scores = pd.DataFrame(index=budget,
                          columns=[str(p) + '-mean' for p in partitions] +
                                  [str(p) + '-std' for p in partitions])

    print
    print 'Testing performance on perturbed data with different partition sizes'
    print

    n_trials = args.num_trials
    n_folds = args.num_folds
    n_parts = args.num_partitions
    n_subsets = args.num_subsets

    for n_parts in partitions:
        print 'Testing perturbation for', n_parts, 'partitions'
        shape = (n_trials, len(budget))
        results = pd.DataFrame(np.zeros(shape), columns=budget)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for b in budget:
            eps = b / float(n_subsets)
            res = test_classifier(classifier=SubsetForest,
                                  df=df,
                                  y=labels,
                                  epsilon=eps,
                                  perturb_type=args.perturb_type,
                                  n_trials=n_trials,
                                  n_folds=n_folds,
                                  n_parts=n_parts,
                                  n_subsets=n_subsets,
                                  subset_size=args.subset_size,
                                  bucket=True,
                                  cols=list(df.columns))

            results[b] = res['auc']
            err = res['auc'].std() / np.sqrt(n_trials)
            print
            print '%d parts, budget = %.2f: %.3f (+- %.3f)' % (
                n_parts, b, res['auc'].mean(), err)
            print

        # aggregate the scores for each trial
        for b in budget:
            mean = results[b].as_matrix().mean()
            std = results[b].as_matrix().std()
            scores.loc[b, '%d-mean' % n_parts] = mean
            scores.loc[b, '%d-std' % n_parts] = std
            print '\tbudget = %.3f: %.3f (+- %.3f)' % (b, mean, std)

        if args.plot:
            plt.errorbar(budget, scores['%d-mean' % n_parts],
                         yerr=scores['%d-std' % n_parts])

    outfile = args.out_file or 'perturbation-num-partitions.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0, 5.0, 0.5, 1.0])
        plt.xlabel('epsilon')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def plot_perturbation_datasets():
    """
    Plot performance of a few different datasets by perturbation
    """
    files = [
        ('edx/3091x_f12/features-all-wk10-ld4.csv', 'edx/edx-feats.txt', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-all-wk10-ld4.csv', 'edx/edx-feats.txt', 'dropout', 'b', '6002x'),
        ('census/features.csv', 'census/census-feats.txt', 'label', 'g', 'census'),
    ]

    # try ten different budgets, with epsilon from 1 to 5
    budget = np.linspace(0.5, 5, 15)
    scores = pd.DataFrame(index=budget, columns=[f[-1] + '-mean' for f in files] +
                                                [f[-1] + '-std' for f in files])

    # pull from the flags for now; these could also be set programatically
    n_folds = args.num_folds
    n_subsets = args.num_subsets
    n_trials = args.num_trials
    n_parts = args.num_partitions

    for f, feats, label, fmt, name in files:
        print
        print 'Testing perturbations on dataset', repr(name)
        print
        df = load_csv('data/' + f, 'features/' + feats)
        labels = df[label].values
        df = df
        del df[label]

        # we're gonna store results from each trial in a big array for now, then
        # compute standard deviation on everything later
        shape = (n_trials, len(budget))
        results = pd.DataFrame(np.zeros(shape), columns=budget)

        for b in budget:
            eps = b / float(n_subsets)
            if args.verbose >= 1:
                print
                print 'budget = %.3f / %d' % (b, n_subsets)

            # dooo itt
            res = test_classifier(classifier=SubsetForest,
                                  df=df,
                                  y=labels,
                                  epsilon=eps,
                                  perturb_type=args.perturb_type,
                                  n_folds=n_folds,
                                  n_trials=n_trials,
                                  n_parts=n_parts,
                                  n_subsets=n_subsets,
                                  subset_size=args.subset_size,
                                  bucket=True,
                                  cols=list(df.columns))

            # results are in the form of an array, so we drop that into its
            # appropriate slice here
            results[b] = res['auc']
            err = res['auc'].std() / np.sqrt(n_trials)
            if args.verbose >= 1:
                print '%s, budget = %.2f: %.3f (+- %.3f)' % (
                    name, b, res['auc'].mean(), err)

        print '%s results:' % name
        # aggregate the scores for each trial
        for b in budget:
            mean = results[b].as_matrix().mean()
            std = results[b].as_matrix().std()
            scores.ix[b, name+'-mean'] = mean
            scores.ix[b, name+'-std'] = std
            print '\tbudget = %.3f: %.3f (+- %.3f)' % (b, mean, std)

        if args.plot:
            plt.errorbar(budget, scores[name+'-mean'], yerr=scores[name+'-std'],
                         fmt=fmt)

    outfile = args.out_file or 'pert-by-dataset.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0.0, 5.0, 0.5, 1.0])
        plt.xlabel('epsilon')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Epsilon, with Standard Error')
        plt.show()


def plot_perturbation_method_privacy():
    """
    Plot performance as a function of number of partitions, for fixed budget.
    """
    df = load_csv(args.data_file, args.feature_file)
    labels = df[args.label].values
    del df[args.label]

    methods = ['pram', 'rappor', 'bits']
    budget = np.linspace(0.5, 10, 19)
    scores = pd.DataFrame(index=budget,
                          columns=[str(m) + '-mean' for m in methods] +
                                  [str(m) + '-std' for m in methods])

    print
    print 'Testing performance on perturbed data with different partition sizes'
    print

    n_trials = args.num_trials
    n_folds = args.num_folds
    n_parts = args.num_partitions
    n_subsets = args.num_subsets
    subset_size = args.subset_size

    for method in methods:
        print 'Testing performance with perturbation method', repr(method)
        shape = (n_trials, len(budget))
        results = pd.DataFrame(np.zeros(shape), columns=budget)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for b in budget:
            eps = b / float(n_subsets)
            res = test_classifier(classifier=SubsetForest,
                                  df=df,
                                  y=labels,
                                  epsilon=eps,
                                  perturb_type=method,
                                  n_trials=n_trials,
                                  n_folds=n_folds,
                                  n_parts=n_parts,
                                  n_subsets=n_subsets,
                                  subset_size=subset_size,
                                  bucket=False,
                                  cols=list(df.columns))

            results[b] = res['auc']
            print
            print 'method %s, budget = %.2f: %.3f (+- %.3f)' % (
                method, b, res['auc'].mean(), res['auc'].std())
            print

        # aggregate the scores for each trial
        print 'Method', method
        for b in budget:
            mean = results[b].as_matrix().mean()
            std = results[b].as_matrix().std()
            scores.loc[b, '%d-mean' % method] = mean
            scores.loc[b, '%d-std' % method] = std
            print '\tbudget = %.3f: %.3f (+- %.3f)' % (b, mean, std)

        if args.plot:
            plt.errorbar(budget, scores['%d-mean' % method],
                         yerr=scores['%d-std' % method],
                         fmt=fmt)

    outfile = args.out_file or 'pert-method.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0, 5.0, 0.5, 1.0])
        plt.xlabel('perturbation')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def plot_perturbation_method_size():
    """
    Plot performance as a function of number of partitions, for fixed budget.
    """
    df = load_csv(args.data_file, args.feature_file)
    labels = df[args.label].values
    del df[args.label]

    methods = ['pram', 'rappor', 'bits']
    sizes = range(1, 7)
    scores = pd.DataFrame(index=budget,
                          columns=[str(m) + '-mean' for m in methods] +
                                  [str(m) + '-std' for m in methods])

    print
    print 'Testing performance on perturbed data with different partition sizes'
    print

    n_trials = args.num_trials
    n_folds = args.num_folds
    n_parts = args.num_partitions
    n_subsets = args.num_subsets
    eps = args.epsilon

    for method in methods:
        print 'Testing performance with perturbation method', repr(method)
        shape = (n_trials, len(sizes))
        results = pd.DataFrame(np.zeros(shape), columns=sizes)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for size in sizes:
            res = test_classifier(classifier=SubsetForest,
                                  df=df,
                                  y=labels,
                                  epsilon=eps,
                                  perturb_type=method,
                                  n_trials=n_trials,
                                  n_folds=n_folds,
                                  n_parts=n_parts,
                                  n_subsets=n_subsets,
                                  subset_size=size,
                                  bucket=False,
                                  cols=list(df.columns))

            results[b] = res['auc']
            print
            print 'method %s, cardinality = %d: %.3f (+- %.3f)' % (
                method, 2**(size+1), res['auc'].mean(), res['auc'].std())
            print

        # aggregate the scores for each trial
        print 'Method', method
        for size in sizes:
            mean = results[size].as_matrix().mean()
            std = results[size].as_matrix().std()
            scores.loc[size, '%d-mean' % method] = mean
            scores.loc[size, '%d-std' % method] = std
            print '\tcardinality = %d: %.3f (+- %.3f)' % (2**(size+1),
                                                          mean, std)

        if args.plot:
            plt.errorbar(sizes, scores['%d-mean' % n_parts],
                         yerr=scores['%d-std' % n_parts],
                         fmt=fmt)

    outfile = args.out_file or 'method-subset-size.csv'
    with open(outfile, 'w') as f:
        scores.to_csv(f)

    if args.plot:
        plt.axis([0, 5.0, 0.5, 1.0])
        plt.xlabel('perturbation')
        plt.ylabel('roc_auc')
        plt.title('AUC vs. Perturbation, with Standard Deviation Error')
        plt.show()


def simple_test():
    """
    Run one test on the subset forest
    required args: data-file, epsilon
    """
    df = load_csv(args.data_file, args.feature_file)
    labels = df[args.label].values
    del df[args.label]

    # load partitions/subsets if they're there
    subsets = load_subsets(args.subset_file)

    # test the silly ensemble
    res = test_classifier(classifier=SubsetForest,
                          df=df, y=labels,
                          subsets=subsets,
                          subset_size=args.subset_size,
                          n_parts=args.num_partitions,
                          n_subsets=args.num_subsets,
                          epsilon=args.epsilon,
                          perturb_type=args.perturb_type,
                          n_trials=args.num_trials,
                          n_folds=args.num_folds,
                          bucket=True,
                          clf_metric=args.clf_metric,
                          cols=list(df.columns))

    print
    for met, arr in res.items():
        # margin of error: two standard deviations around the mean
        moe = 2 * arr.std() / np.sqrt(args.num_trials)
        print met, 'mean: %.4f (+- %.4f)' % (arr.mean(), moe)


def main():
    for test in args.tests:
        if test == 'compare-classifiers':
            compare_classifiers()
        if test == 'subset-size-datasets':
            plot_subset_size_datasets()
        if test == 'binning-datasets':
            plot_binning_datasets()
        if test == 'perturbation-subset-size':
            plot_perturbation_subset_size()
        if test == 'perturbation-datasets':
            plot_perturbation_datasets()
        if test == 'perturbation-partitions':
            plot_perturbation_partitions()
        if test == 'perturbation-method':
            plot_perturbation_method_privacy()
        if test == 'method-subset-size':
            plot_perturbation_method_size()
        if test == 'simple':
            simple_test()


# TODO: plot p = 1-q vs random response vs optimal p, q
#   plot measured error vs performance
#   plot histograms of best decision trees
#   plot partitioning vs performance

if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
    print sorted(global_total.items(), key=lambda i: -i[1])
