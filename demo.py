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

ap = argparse.ArgumentParser()
ap.add_argument('data_file', type=str, help='path to the raw data file')
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
ap.add_argument('--num-folds', type=int, default=5,
                help='number of folds on which to test each classifier')


def test_classifier(classifier, frame, y, perturb=0, n_folds=5,
                    verbose=0, **kwargs):
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
        print '\tf1: mean = %f, std = %f' % (np_f1.mean(), np_f1.std())
        print '\tAUC: mean = %f, std = %f' % (np_auc.mean(), np_auc.std())
        print '\tAccuracy: mean = %f, std = %f' % (np_acc.mean(), np_acc.std())

    return clf, np_auc


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

    #plt.plot(range(df.shape[0]), df['gen_distance'])
    #plt.plot(range(ndf.shape[0]), ndf['gen_distance'], 'ro')
    #plt.title('Normal vs. Perturbed')
    #plt.show()

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


def compare_classifiers(df):
    """
    Run a bunch of different classifiers on one dataset and print the results
    """
    labels = df[args.label].values
    del df[args.label]

    subsets = []
    if args.subsets:
        with open(args.subsets) as f:
            for l in f:
                subsets.append([c.strip() for c in l.split(',')])
    else:
        subsets = generate_subsets(df, args.num_subsets, args.subset_size)

    # test a Random Forest classifier, the gold standard.
    test_classifier(classifier=RandomForestClassifier, frame=df, y=labels,
                    n_folds=args.num_folds, class_weight='balanced')

    # Adaboost
    test_classifier(classifier=AdaBoostClassifier, frame=df, y=labels,
                    n_folds=args.num_folds)

    # gradient boosting
    test_classifier(classifier=GradientBoostingClassifier, frame=df, y=labels,
                    n_folds=args.num_folds)

    # test BaggingClassifier: very similar to our classifier; uses random
    # subsets of features to build decision trees
    #test_classifier(classifier=BaggingClassifier, frame=df, y=labels,
                    #n_folds=args.num_folds, #max_features=args.subset_size,
                    #base_estimator=sktree.DecisionTreeClassifier(
                        #class_weight='balanced'))

    # test our weird whatever
    clf, npres = test_classifier(classifier=SubspaceForest, frame=df,
                                 y=labels, n_folds=args.num_folds,
                                 perturb=args.perturbation, df=df,
                                 labels=labels, subsets=subsets)

    print
    print 'Top scoring features for last SubspaceForest classifier:'
    clf.print_scores()

    with open('last-features.txt', 'w') as f:
        for ss, cols in clf.cols.iteritems():
            f.write(','.join(cols) + '\n')


def get_perturbation(df, labels, x, subsets):
    """
    Calculate the performance of a classifier for every perturbation level in
    {0, 0.1, ..., 0.9}
    """
    y = []
    yerr = []

    for pert in x:
        clf, res = test_classifier(classifier=SubspaceForest, frame=df,
                                   y=labels, perturb=pert,
                                   n_folds=args.num_folds, df=df,
                                   labels=labels, subsets=subsets)

        y.append(res.mean())
        yerr.append(res.std())
        print 'p = %.3f: %.3f (+- %.3f)' % (pert, res.mean(), res.std())

    return y, yerr


def plot_subset_size_of_datasets():
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
    scores = pd.DataFrame(index=x, columns=[f[0] for f in files])
    stdevs = pd.DataFrame(index=x, columns=[f[0] for f in files])

    for f, label, shape in files:
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
            clf, res = test_classifier(classifier=SubspaceForest, frame=df,
                                       y=labels, perturb=0,
                                       n_folds=args.num_folds, df=df,
                                       labels=labels, subsets=subsets)
            y.append(res.mean())
            yerr.append(res.std())

            scores.set_value(subset_size, f, res.mean())
            stdevs.set_value(subset_size, f, res.std())

        plt.errorbar(x, y, yerr=yerr, fmt=shape)

    with open('subset-size-of-datasets-means.csv', 'w') as f:
        scores.to_csv(f)
    with open('subset-size-of-datasets-errs.csv', 'w') as f:
        stdevs.to_csv(f)

    plt.axis([0.5, biggest_subset + 0.5, 0.5, 1.0])
    plt.xlabel('subset size')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Subset Size, with Standard Deviation Error')
    plt.show()


def plot_perturbation_of_subset_size(df):
    labels = df[args.label].values
    del df[args.label]

    biggest_subset = 5
    pairs = zip(range(1, biggest_subset + 1), ['r', 'b', 'g', 'y', 'k'])
    x = [float(i)/10.0 for i in range(10)] + [.92, .94, .96, .98, 1.0]

    scores = pd.DataFrame(index=x, columns=[p[0] for p in pairs])
    stdevs = pd.DataFrame(index=x, columns=[p[0] for p in pairs])

    for i, shape in pairs:
        subsets = generate_subsets(df, -1, subset_size)
        y, yerr = get_perturbation(df, labels, x, subsets)
        scores[i] = y
        stdevs[i] = yerr
        plt.errorbar(x, y, yerr=yerr)

    with open('perturbation-ss-means.csv', 'w') as f:
        scores.to_csv(f)
    with open('perturbation-ss-errs.csv', 'w') as f:
        stdevs.to_csv(f)

    plt.axis([0.0, 1.0, 0.5, 1.0])
    plt.xlabel('perturbation')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Perturbation, with Standard Deviation Error')
    plt.show()


def plot_perturbation_of_datasets():
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
    scores = pd.DataFrame(index=x, columns=[f[-1] for f in files])
    stdevs = pd.DataFrame(index=x, columns=[f[-1] for f in files])

    baboon_subsets = []
    with open('baboon_mating/subsets.txt') as f:
        for l in f:
            baboon_subsets.append([c.strip() for c in l.split(',')])

    for f, label, shape, name in files:
        df = pd.read_csv(f)
        labels = df[label].values
        del df[label]
        subsets = generate_subsets(df, -1, args.subset_size)
        #if name == 'baboon-mating':
            #subsets = baboon_subsets

        print
        print 'Testing perturbations on dataset', repr(name)

        y, yerr = get_perturbation(df, labels, x, subsets)
        scores[name] = y
        stdevs[name] = yerr
        plt.errorbar(x, y, yerr=yerr, fmt=shape)

    with open('pert-auc-means.csv', 'w') as f:
        scores.to_csv(f)
    with open('pert-auc-errs.csv', 'w') as f:
        stdevs.to_csv(f)

    plt.axis([0.0, 1.0, 0.5, 1.0])
    plt.xlabel('perturbation')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Perturbation, with Standard Deviation Error')
    plt.show()


def plot_binning_of_datasets():
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
    scores = pd.DataFrame(index=x, columns=[f[-1] for f in files])
    stdevs = pd.DataFrame(index=x, columns=[f[-1] for f in files])

    baboon_subsets = []
    with open('baboon_mating/subsets.txt') as f:
        for l in f:
            baboon_subsets.append([c.strip() for c in l.split(',')])

    for f, label, shape, name in files:
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

            clf, res = test_classifier(classifier=SubspaceForest, frame=df,
                                       y=labels, perturb=0,
                                       n_folds=args.num_folds, df=df,
                                       labels=labels, subsets=subsets)
            y.append(res.mean())
            yerr.append(res.std())

            scores.set_value(bin_size, name, res.mean())
            stdevs.set_value(bin_size, name, res.std())

        plt.errorbar(x, y, yerr=yerr, fmt=shape)

    with open('binsize-auc-means.csv', 'w') as f:
        scores.to_csv(f)
    with open('binsize-auc-errs.csv', 'w') as f:
        stdevs.to_csv(f)

    plt.axis([-2, 22, 0.5, 1.0])
    plt.xlabel('subset size')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Subset Size, with Standard Deviation Error')
    plt.show()


def main():
    #df = pd.read_csv(open(args.data_file))
    #compare_classifiers(df)
    #plot_subset_size_of_datasets()
    plot_perturbation_of_datasets()
    #plot_binning_of_datasets()


if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
