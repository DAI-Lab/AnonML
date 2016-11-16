import pandas as pd
import numpy as np
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
ap.add_argument('--subset-size', type=int, default=4,
                help='number of features per generated subset')
ap.add_argument('--perturbation', type=float, default=0,
                help="probability of perturbation")
ap.add_argument('--subsets', type=str, default=None,
                help='hard-coded subset file')
args = ap.parse_args()


def test_classifier(classifier, X, y, perturb=False, n_folds=3, **kwargs):
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
            X_train = perturb_data(X_train)

        clf.fit(X_train, y_train)

        # score the superclassifier
        scorer = check_scoring(clf, scoring='roc_auc')
        results.append(scorer(clf, X_test, y_test))

    npres = np.array(results)
    print 'Results (%s): mean = %f, std = %f' % (classifier.__name__,
                                                 npres.mean(), npres.std())

def perturb_data(X):
    if args.perturbation == 0:
        return

    print "Perturbing data with factor", args.perturbation
    # for each value in the dataframe, with 1 - args.perturbation probability,
    # switch the value a random bucket.
    max_val = {}
    for col, val in enumerate(X[0]):
        if val.dtype == 'int64':
            max_val[col] = int(np.max(X[:, col])) + 1

    nX = X.copy()
    for i in xrange(nX.shape[0]):
        for j in max_val:
            val = nX[i, j]
            if random.random() < args.perturbation:
                val = random.choice(range(max_val[j]))
            nX[i, j] = val

    return nX


def main():
    df = pd.read_csv(open(args.data_file))
    labels = df[args.label].values
    del df[args.label]

    subsets = []
    if args.subsets:
        with open(args.subsets) as f:
            for l in f:
                subsets.append([c.strip() for c in l.split(',')])

    test_classifier(classifier=BaggingClassifier, X=df.as_matrix(), y=labels,
                    base_estimator=sktree.DecisionTreeClassifier(),
                    max_features=4)
    test_classifier(classifier=RandomForestClassifier, X=df.as_matrix(),
                    y=labels)
    test_classifier(classifier=SubsetForest, X=df.as_matrix(), y=labels,
                    perturb=True, n_folds=10, df=df, labels=labels,
                    subsets=subsets or None, n_subsets=args.num_subsets,
                    subset_size=args.subset_size)


if __name__ == '__main__':
    main()
