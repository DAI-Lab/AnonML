import pandas as pd
import numpy as np
import argparse
import pdb
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
args = ap.parse_args()


def test_classifier(classifier, X, y, **kwargs):
    X = np.nan_to_num(X)
    y = np.array(y)
    folds = KFold(y.shape[0], n_folds=3, shuffle=True)
    results = []
    clf = classifier(**kwargs)
    for train_index, test_index in folds:
        # make 3 folds of the data for training
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        # score the superclassifier
        scorer = check_scoring(clf, scoring='roc_auc')
        results.append(scorer(clf, X_test, y_test))

    npres = np.array(results)
    print 'Results (%s): mean = %f, std = %f' % (classifier.__name__,
                                                 npres.mean(), npres.std())


def main():
    df = pd.read_csv(open(args.data_file))
    labels = df[args.label].values
    del df[args.label]

    test_classifier(classifier=BaggingClassifier, X=df.as_matrix(), y=labels,
                    base_estimator=sktree.DecisionTreeClassifier(), max_features=4)
    test_classifier(classifier=RandomForestClassifier, X=df.as_matrix(),
                    y=labels)
    test_classifier(classifier=SubsetForest, X=df.as_matrix(), y=labels, df=df,
                    labels=labels, subsets=None, n_subsets=args.num_subsets,
                    subset_size=args.subset_size)


if __name__ == '__main__':
    main()
