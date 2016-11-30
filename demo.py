#!/usr/bin/python2.7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import random

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
from subset_forest import SubsetForest

ap = argparse.ArgumentParser()
ap.add_argument('data_file', type=str, help='path to the raw data file')
ap.add_argument('--label', type=str, default='dropout',
                help='label we are trying to predict')
ap.add_argument('--perturbation', type=float, default=0,
                help="probability of perturbation")
ap.add_argument('--subsets', type=str, default=None,
                help='hard-coded subset file')
ap.add_argument('--num-subsets', type=int, default=20,
                help='number of subsets to generate')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--num-folds', type=int, default=10,
                help='number of folds on which to test each classifier')


def test_classifier(classifier, frame, y, perturb=0, n_folds=10, **kwargs):
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
        #print
        #print '\tPredicted/actual true:', sum(y_pred), sum(y_test)
        #print '\tPredicted/actual false:', sum(~y_pred), sum(~y_test)
        tp, tn = sum(y_pred & y_test), sum(~y_pred & ~y_test)
        fp, fn = sum(y_pred & ~y_test), sum(~y_pred & y_test)
        #print '\tTrue positive (rate): %d, %.3f' % (tp, float(tp) / sum(y_test))
        #print '\tTrue negative (rate): %d, %.3f' % (tn, float(tn) / sum(~y_test))
        #print '\tFalse positive (rate): %d, %.3f' % (fp, float(fp) / sum(~y_test))
        #print '\tFalse negative (rate): %d, %.3f' % (fn, float(fn) / sum(y_test))

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
    print 'Results (%s, %d trials):' % (classifier.__name__, n_folds)
    print '\tf1: mean = %f, std = %f' % (np_f1.mean(), np_f1.std())
    print '\tAUC: mean = %f, std = %f' % (np_auc.mean(), np_auc.std())
    print '\tAccuracy: mean = %f, std = %f' % (np_acc.mean(), np_acc.std())
    return clf, np_auc


def perturb_dataframe(df, perturbation, subsets=None):
    """
    For each row in the dataframe, for each subset of that row, randomly perturb
    all the values.
    """
    if subsets is None:
        subsets = [[i] for i in df.columns]

    # for each value in the dataframe, with 1 - perturbation probability,
    # switch the value a random bucket.
    perturb_vals = {}
    for col in df.columns:
        series = df[col]
        if series.dtype == 'int64' or series.dtype == 'object':
            perturb_vals[col] = ('discrete', int(np.min(series)),
                                 int(np.max(series)))
        if series.dtype == 'float64':
            series = series[~np.isnan(series)]
            diffs = np.diff(np.sort(series))
            min_diff = np.min(diffs[np.nonzero(diffs)])
            perturb_vals[col] = ('continuous', np.min(series),
                                 np.max(series), min_diff)

    ndf = df.copy()
    for i, row in df.iterrows():
        for cols in subsets:
            if random.random() < perturbation:
                for col in cols:
                    pert = perturb_vals[col]
                    val = row[col]
                    if pert[0] == 'discrete':
                        val = random.choice(range(pert[1], pert[2]))
                    if pert[0] == 'continuous':
                        rng = pert[2] - pert[1]
                        # random value btwn min and max
                        val = random.random() * rng + pert[1]
                    ndf.set_value(i, col, val)

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

    ## Logistic regression
    #test_classifier(classifier=LogisticRegression, frame=df, y=labels,
                    #n_folds=args.num_folds, C=1.0)

    ## 5 nearest neighbors
    #test_classifier(classifier=KNeighborsClassifier, frame=df, y=labels,
                    #n_folds=args.num_folds, n_neighbors=5)

    # test BaggingClassifier: very similar to our classifier; uses random
    # subsets of features to build decision trees
    test_classifier(classifier=BaggingClassifier, frame=df, y=labels,
                    n_folds=args.num_folds, max_features=args.subset_size,
                    base_estimator=sktree.DecisionTreeClassifier(
                        class_weight='balanced'))

    # test our weird whatever
    clf, npres = test_classifier(classifier=SubsetForest, frame=df,
                                 y=labels, n_folds=args.num_folds,
                                 perturb=args.perturbation, df=df,
                                 labels=labels, subsets=subsets)

    print
    print 'Top scoring features for last SubsetForest classifier:'
    clf.print_scores()

    with open('last-features.txt', 'w') as f:
        for ss, cols in clf.cols.iteritems():
            f.write(','.join(cols) + '\n')


def get_perturbation(df, subsets):
    """
    Calculate the performance of a classifier for every perturbation level in
    {0, 0.1, ..., 0.9}
    """
    x = [float(i)/10.0 for i in range(10)] #+ [.92, .94, .96, .98, .99]
    y = []
    yerr = []

    for pert in x:
        clf, res = test_classifier(classifier=SubsetForest, frame=df,
                                   y=labels, perturb=pert,
                                   n_folds=args.num_folds, df=df,
                                   labels=labels, subsets=subsets)

        y.append(res.mean())
        yerr.append(res.std())

    return x, y, yerr


def plot_subset_size_of_datasets():
    """
    Plot performance of a few different datasets across a number of different
    subset sizes
    """
    files = {}
    biggest_subset = 5
    x = range(biggest_subset)

    for f, shape in files:
        df = pd.read_csv(f)
        y = []
        yerr = []
        for subset_size in x:
            clf, res = test_classifier(classifier=SubsetForest, frame=df,
                                       y=labels, perturb=pert,
                                       n_folds=args.num_folds, df=df,
                                       labels=labels, subsets=subsets)
            y.append(res.mean())
            yerr.append(res.std())

        plt.errorbar(x, y, shape, yerr=yerr)

    plt.axis([0.0, 1.0, 0.5, 1.0])
    plt.xlabel('subset size')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Subset Size, with Standard Deviation Error')
    plt.show()


def plot_perturbation_of_subset_size(df):
    labels = df[args.label].values
    del df[args.label]

    biggest_subset = 5
    for i in range(1, biggest_subset + 1):
        subsets = generate_subsets(df, -1, 1)

    plt.errorbar(x, y, yerr=yerr)
    plt.axis([0.0, 1.0, 0.5, 1.0])
    plt.xlabel('perturbation')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Perturbation, with Standard Deviation Error')
    plt.show()


def main():
    df = pd.read_csv(open(args.data_file))
    compare_classifiers(df)
    #plot_perturbations(df)


if __name__ == '__main__':
    global args
    args = ap.parse_args()
    main()
