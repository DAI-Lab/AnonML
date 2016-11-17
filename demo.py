import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import random
from sklearn import tree as sktree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring
from subset_forest import SubsetForest

ap = argparse.ArgumentParser()
ap.add_argument('data_file', type=str, help='path to the raw data file')
ap.add_argument('--label', type=str, default='dropout',
                help='label we are trying to predict')
ap.add_argument('--num-subsets', type=int, default=20,
                help='number of subsets to generate')
ap.add_argument('--subset-size', type=int, default=3,
                help='number of features per generated subset')
ap.add_argument('--perturbation', type=float, default=0,
                help="probability of perturbation")
ap.add_argument('--subsets', type=str, default=None,
                help='hard-coded subset file')


def test_classifier(classifier, X, y, perturb=0, n_folds=10, **kwargs):
    X = np.nan_to_num(X)
    y = np.array(y)
    folds = KFold(y.shape[0], n_folds=n_folds, shuffle=True)
    results = []
    clf = classifier(**kwargs)
    for train_index, test_index in folds:
        # make 3 folds of the data for training
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # perturb the data if necessary
        if perturb:
            X_train = perturb_data(X_train, perturb)

        clf.fit(X_train, y_train)

        # score the superclassifier
        scorer = check_scoring(clf, scoring='roc_auc')
        results.append(scorer(clf, X_test, y_test))

    npres = np.array(results)
    print 'Results (%s, %d trials): mean = %f, std = %f' % \
        (classifier.__name__, n_folds, npres.mean(), npres.std())
    return clf, npres


def perturb_data(X, perturbation):
    # for each value in the dataframe, with 1 - perturbation probability,
    # switch the value a random bucket.
    perturb_vals = {}
    for col, val in enumerate(X[0]):
        series = X[:, col]
        print series.dtype, col
        if series.dtype == 'int64' or series.dtype == 'object':
            perturb_vals[col] = ('discrete', int(np.min(series)),
                                 int(np.max(series)))
        if series.dtype == 'float64':
            diffs = np.diff(np.sort(series))
            min_diff = np.min(diffs[np.nonzero(diffs)])
            perturb_vals[col] = ('continuous', np.min(series),
                                 np.max(series), min_diff)

    nX = X.copy()
    for i in xrange(nX.shape[0]):
        for j, pert in perturb_vals.iteritems():
            val = nX[i, j]
            if pert[0] == 'discrete':
                if random.random() < perturbation:
                    val = random.choice(range(pert[1], pert[2]))
            if pert[0] == 'continuous':
                rng = pert[2] - pert[1]
                # random value btwn min and max
                if random.random() < perturbation:
                    val = random.random() * rng + pert[1]
                else:
                    noise = np.random.laplace(0.0, pert[3]/8)
                    val += noise
            nX[i, j] = val

    return nX


def compare_classifiers(df):
    labels = df[args.label].values
    del df[args.label]

    #for col in df.columns:
        #if df[col].dtype == 'int64':
            #df[col] = df[col].astype('object')

    matrix = df.to_records(index=False)
    matrix.reshape(matrix.shape[1], matrix.shape[0])
    print type(matrix)
    print matrix.dtype

    subsets = []
    if args.subsets:
        with open(args.subsets) as f:
            for l in f:
                subsets.append([c.strip() for c in l.split(',')])

    test_classifier(classifier=BaggingClassifier, X=matrix, y=labels,
                    base_estimator=sktree.DecisionTreeClassifier(),
                    max_features=4)
    test_classifier(classifier=RandomForestClassifier, X=matrix,
                    y=labels)
    clf, npres = test_classifier(classifier=SubsetForest, X=matrix,
                                 y=labels, perturb=args.perturbation, df=df,
                                 labels=labels, subsets=subsets or None,
                                 n_subsets=args.num_subsets,
                                 subset_size=args.subset_size)

    print
    print 'Top scoring features for last SubsetForest classifier:'
    clf.print_scores()

    with open('last-features.txt', 'w') as f:
        for ss, cols in clf.cols.iteritems():
            f.write(','.join(cols) + '\n')


def plot_perturbation(df):
    labels = df[args.label].values
    del df[args.label]

    subsets = []
    if args.subsets:
        with open(args.subsets) as f:
            for l in f:
                subsets.append([c.strip() for c in l.split(',')])

    x = [float(i)/10.0 for i in range(10)] #+ [.92, .94, .96, .98, .99]
    y = []
    yerr = []

    for pert in x:
        clf, res = test_classifier(classifier=SubsetForest, X=df.as_matrix(),
                                   y=labels, perturb=pert, n_folds=5, df=df,
                                   labels=labels, subsets=subsets or None,
                                   n_subsets=args.num_subsets,
                                   subset_size=args.subset_size)

        y.append(res.mean())
        yerr.append(res.std())

    plt.errorbar(x, y, yerr=yerr)
    plt.axis([0.0, 1.0, 0.5, 1.0])
    plt.xlabel('Perturbation')
    plt.ylabel('roc_auc')
    plt.title('AUC vs. Perturbation, with Standard Deviation Error')
    plt.show()


def main():
    global args
    args = ap.parse_args()
    df = pd.read_csv(open(args.data_file))
    compare_classifiers(df)
    #plot_perturbation(df)


if __name__ == '__main__':
    main()
