#!/usr/bin/python2.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import random
import itertools

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
from subset_forest import SubspaceForest


TEST_TYPES = ['compare-classifiers', 'subset-size-datasets', 'perturbation-subset-size', 'perturbation-datasets', 'binning-datasets']

ap = argparse.ArgumentParser()
ap.add_argument('tests', type=str, nargs='+', choices=TEST_TYPES,
                help='name of test to run')
ap.add_argument('--data-file', type=str, help='path to the raw data file')
ap.add_argument('--plot', action='store_true',
                help='whether to plot the results of the test')
ap.add_argument('--label', type=str, default='dropout',
                help='label we are trying to predict')
ap.add_argument('--perturbation', type=float, default=0,
                help="probability of perturbation")
ap.add_argument('--subsets', type=str, default=None,
                help='hard-coded subset file')
ap.add_argument('--num-subsets', type=int, default=-1,
                help='number of subsets to generate; -1 == all')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--recursive-subsets', action='store_true',
                help='generates all subsets that fit in the largest subset')
ap.add_argument('--num-trials', type=int, default=1,
                help='number of times to try with different subsets')
ap.add_argument('--num-folds', type=int, default=5,
                help='number of folds on which to test each classifier')


###############################################################################
##  Misc helper functions  #########################################################
###############################################################################

def perturb_dataframe(df, perturbation, subsets=None):
    """
    For each row in the dataframe, for each subset of that row, randomly perturb
    all the values of that subset.
    """
    if subsets is None:
        subsets = [[i] for i in df.columns]

    ndf = df.copy()
    for cols in subsets:
        ix = df.index.to_series().sample(frac=perturbation)
        for col in cols:
            ndf.ix[ix, col] = np.random.choice(df[col], size=len(ix))

    return ndf


def generate_subsets(df, n_subsets, subset_size):
    """
    Generate n_subsets random, non-overlapping subsets of subset_size columns each
    """
    shuf_cols = list(df.columns)
    random.shuffle(shuf_cols)
    subsets = []
    if n_subsets < 0:
        n_subsets = len(shuf_cols)

    for i in range(n_subsets):
        if not shuf_cols:
            break
        cols = shuf_cols[:subset_size]
        shuf_cols = shuf_cols[subset_size:]

        subsets.append(cols)
        if args.recursive_subsets:
            for j in range(1, subset_size):
                for c in itertools.combinations(cols, j):
                    subsets.append(c)

    return subsets


###############################################################################
##  Test helper functions  #########################################################
###############################################################################

def test_classifier(classifier, frame, y, perturb=0, n_folds=5,
                    verbose=1, **kwargs):
    """
    Run the given classifier with the given perturbation for n_folds tests, and
    return the results.
    """
    X = np.nan_to_num(frame.as_matrix())
    y = np.array(y).astype('bool')
    clf = classifier(**kwargs)
    auc_results = []
    f1_results = []
    acc_results = []

    folds = KFold(y.shape[0], n_folds=n_folds, shuffle=True)
    for train_index, test_index in folds:
        # perturb the data if necessary
        if perturb:
            pframe = perturb_dataframe(frame, perturb,
                                       subsets=kwargs.get('subsets'))
            X_pert = np.nan_to_num(pframe.as_matrix())
        else:
            X_pert = X

        # make 3 folds of the data for training
        X_train, X_test = X_pert[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        tp, tn = sum(y_pred & y_test), sum(~y_pred & ~y_test)
        fp, fn = sum(y_pred & ~y_test), sum(~y_pred & y_test)

        if verbose >= 2:
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
        scorer = check_scoring(clf, scoring='roc_auc')
        auc_results.append(scorer(clf, X_test, y_test))

        scorer = check_scoring(clf, scoring='f1')
        f1_results.append(scorer(clf, X_test, y_test))

        scorer = check_scoring(clf, scoring='accuracy')
        acc_results.append(scorer(clf, X_test, y_test))

    np_f1 = np.array(f1_results)
    np_auc = np.array(auc_results)
    np_acc = np.array(acc_results)
    if verbose >= 1:
        print 'Results (%s, %d trials):' % (classifier.__name__, n_folds)
        print '\tf1: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_f1.mean(), np_f1.std(), np_f1.min(), np_f1.max())
        print '\tAUC: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_auc.mean(), np_auc.std(), np_auc.min(), np_auc.max())
        print '\tAccuracy: mean = %f, std = %f, (%.3f, %.3f)' % \
            (np_acc.mean(), np_acc.std(), np_acc.min(), np_acc.max())

    return clf, {'f1': np_f1, 'auc': np_auc, 'acc': np_acc}


def test_subset_forest(df, labels, perturb=0, n_trials=1, n_folds=5,
                       num_subsets=-1, subset_size=3, subsets=None, verbose=0):
    subsets = subsets or generate_subsets(df, num_subsets, subset_size)
    results = {met: np.ndarray(0) for met in ['f1', 'auc', 'acc']}
    classifiers = []

    for i in range(n_trials):
        clf, res = test_classifier(classifier=SubspaceForest, frame=df,
                                   y=labels, n_folds=n_folds,
                                   perturb=perturb, df=df,
                                   labels=labels, subsets=subsets)
        for met, arr in res.items():
            np.append(results[met], arr)
        classifiers.append((res['auc'].mean(), clf))
        subsets = generate_subsets(df, num_subsets, subset_size)

    return sorted(classifiers)[::-1], results


def test_perturbation(df, labels, x, subsets, n_folds):
    """
    Calculate the performance of a classifier for every perturbation level in
    {0, 0.1, ..., 0.9}
    """
    y = []
    yerr = []

    for pert in x:
        clf, res = test_classifier(classifier=SubspaceForest, frame=df,
                                   y=labels, perturb=pert,
                                   n_folds=n_folds, df=df,
                                   labels=labels, subsets=subsets)

        mean = res['auc'].mean()
        std = res['auc'].std()
        y.append(mean)
        yerr.append(std)
        print 'p = %.3f: %.3f (+- %.3f)' % (pert, mean, std)

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

    total_folds = args.num_folds * args.num_trials

    classifiers = ['random-forest', 'gradient-boost', 'adaboost',
                   'subspace-forest']
    scores = pd.DataFrame(index=classifiers, columns=columns)

    # test our weird whatever
    clfs, res = test_subset_forest(df=df, labels=labels,
                                   perturb=args.perturbation,
                                   n_trials=args.num_trials,
                                   n_folds=args.num_folds,
                                   num_subsets=args.num_subsets,
                                   subset_size=args.subset_size,
                                   subsets=subsets)

    for met, arr in res.items():
        scores.set_value('subspace-forest', met + '-mean', arr.mean())
        scores.set_value('subspace-forest', met + '-std', arr.std())

    print
    print 'Top scoring features for best SubspaceForest classifier:'
    best_clf = clfs[0][1]
    best_clf.print_scores()

    with open('last-features.txt', 'w') as f:
        for ss, cols in best_clf.cols.iteritems():
            f.write(','.join(cols) + '\n')

    # test a Random Forest classifier, the gold standard.
    _, res = test_classifier(classifier=RandomForestClassifier, frame=df,
                             y=labels, n_folds=total_folds,
                             class_weight='balanced')

    for met, arr in res.items():
        scores.set_value('random-forest', met + '-mean', arr.mean())
        scores.set_value('random-forest', met + '-std', arr.std())

    # Adaboost
    _, res = test_classifier(classifier=AdaBoostClassifier, frame=df, y=labels,
                             n_folds=total_folds)

    for met, arr in res.items():
        scores.set_value('adaboost', met + '-mean', arr.mean())
        scores.set_value('adaboost', met + '-std', arr.std())

    # gradient boosting
    _, res = test_classifier(classifier=GradientBoostingClassifier, frame=df,
                             y=labels, n_folds=total_folds)

    for met, arr in res.items():
        scores.set_value('gradient-boost', met + '-mean', arr.mean())
        scores.set_value('gradient-boost', met + '-std', arr.std())

    # test BaggingClassifier: very similar to our classifier; uses random
    # subsets of features to build decision trees
    _, res = test_classifier(classifier=BaggingClassifier, frame=df, y=labels,
                             n_folds=total_folds,
                             #max_features=args.subset_size,
                             base_estimator=sktree.DecisionTreeClassifier(
                                 class_weight='balanced'))

    with open('classifier-comparison.csv', 'w') as f:
        scores.to_csv(f)


def plot_subset_size_datasets(n_folds):
    """
    Plot performance of a few different datasets across a number of different
    subset sizes
    """
    files = [
        ('edx/3091x_f12/features-wk10-ld4-bin.csv', 'dropout', 'r'),
        ('edx/6002x_f12/features-wk10-ld4.csv', 'dropout', 'b'),
        ('edx/201x_sp13/features-wk10-ld4.csv', 'dropout', 'g'),
        ('baboon_mating/raw-features.csv', 'consort', 'k'),
    ]
    biggest_subset = 6
    x = range(1, biggest_subset + 1)
    scores = pd.DataFrame(index=x, columns=[f[0] + '-mean' for f in files] +
                                           [f[0] + '-std' for f in files])

    for f, label, fmt in files:
        df = pd.read_csv(f)
        labels = df[label].values
        del df[label]

        print
        print 'Testing different subset sizes on dataset', f
        print

        y = []
        yerr = []
        for subset_size in x:
            subsets = generate_subsets(df, -1, subset_size)

            _, res = test_subset_forest(df=df, labels=labels, subsets=subsets,
                                        perturb=0, n_trials=args.num_trials,
                                        n_folds=args.num_folds)

            mean = res['auc'].mean()
            std = res['auc'].std()
            y.append(mean)
            yerr.append(std)

            scores.set_value(subset_size, f + '-mean', mean)
            scores.set_value(subset_size, f + '-std', std)

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
    x = [float(i)/10.0 for i in range(10)] + [.95, .98, 1.0]

    scores = pd.DataFrame(index=x, columns=[str(p[0]) + '-mean' for p in pairs] +
                                           [str(p[0]) + '-std' for p in pairs])

    print
    print 'Testing performance on perturbed data with different subspace sizes'
    print

    for subset_size, fmt in pairs:
        print 'Testing perturbation for subset size', subset_size
        results = pd.DataFrame(np.zeros((args.num_folds * 3, len(x))), columns=x)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for i in range(args.num_folds):
            print '\ttesting subspace permutation %d/%d, %d trials each' % \
                (i+1, args.num_folds, 3)
            subsets = generate_subsets(df, -1, subset_size)

            for pert in x:
                _, res = test_classifier(classifier=SubspaceForest, frame=df,
                                         y=labels, perturb=pert,
                                         n_folds=3, df=df,
                                         labels=labels, subsets=subsets)
                results.ix[i*3:i*3+2, pert] = res['auc']
                print '\t\tp = %.2f: %.3f (+- %.3f)' % (pert, res['auc'].mean(),
                                                        res['auc'].std())

        # aggregate the scores for each trial
        for p in x:
            mean = results[p].as_matrix().mean()
            std = results[p].as_matrix().std()
            scores.ix[p, '%d-mean' % i] = mean
            scores.ix[p, '%d-std' % i] = std
            print '\tp = %.3f: %.3f (+- %.3f)' % (p, mean, std)

        if args.plot:
            plt.errorbar(x, scores['%d-mean' % i], yerr=scores['%d-std' % i],
                         fmt=fmt)

    with open('pert-by-subset-size.csv', 'w') as f:
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
        ('baboon_mating/features-b10.csv', 'consort', 'g', 'baboon-mating'),
        ('gender/free-sample-b10.csv', 'class', 'k', 'gender'),
        ('edx/3091x_f12/features-wk10-ld4-b10.csv', 'dropout', 'r', '3091x'),
        ('edx/6002x_f12/features-wk10-ld4-b10.csv', 'dropout', 'b', '6002x'),
    ]
    x = [float(i)/10 for i in range(10)] + [.92, .94, .96, .98, 1.0]
    scores = pd.DataFrame(index=x, columns=[f[-1] + '-mean' for f in files] +
                                           [f[-1] + '-std' for f in files])
    #best_subsets = {f[-1]: (0,) for f in files}

    for f, label, fmt, name in files:
        print
        print 'Testing perturbations on dataset', repr(name)
        df = pd.read_csv(f)
        labels = df[label].values
        del df[label]

        results = pd.DataFrame(np.zeros((args.num_folds * 3, len(x))), columns=x)

        # try each perturbation level with several different subspaces, but keep
        # those subspaces consistent
        for i in range(args.num_folds):
            print '\ttesting subspace permutation %d/%d on %s, %d trials each' % \
                (i+1, args.num_folds, name, 3)
            subsets = generate_subsets(df, -1, args.subset_size)

            for pert in x:
                _, res = test_classifier(classifier=SubspaceForest, frame=df,
                                         y=labels, perturb=pert,
                                         n_folds=3, df=df,
                                         labels=labels, subsets=subsets)
                results.ix[i*3:i*3+2, pert] = res['auc']
                print '\t\tp = %.2f: %.3f (+- %.3f)' % (pert, res['auc'].mean(),
                                                        res['auc'].std())


        print '\t%s results:' % name
        # aggregate the scores for each trial
        for p in x:
            mean = results[p].as_matrix().mean()
            std = results[p].as_matrix().std()
            scores.ix[p, name+'-mean'] = mean
            scores.ix[p, name+'-std'] = std
            print '\tp = %.3f: %.3f (+- %.3f)' % (p, mean, std)

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
            df = pd.read_csv(f % extra)
            labels = df[label].values
            del df[label]

            print 'loaded dataset', f % extra

            if not subsets:
                subsets = generate_subsets(df, -1, args.subset_size)

            _, res = test_subset_forest(df=df, labels=labels,
                                        n_trials=args.num_trials,
                                        n_folds=args.num_folds,
                                        num_subsets=args.num_subsets,
                                        subset_size=args.subset_size,
                                        subsets=subsets, perturb=0)
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


if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
