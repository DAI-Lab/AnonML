import numpy as np
import pdb
from random import shuffle
from sklearn import tree as sktree
from sklearn.ensemble.forest import ForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring

class SubsetForest(ForestClassifier):
    def __init__(self, df, labels, subsets=None, verbose=False):
        """
        df: dataframe of training data (no label column)
        labels: series of boolean labels
        subsets: list of lists of column names in the dataframe
        """
        self.labels = labels
        self.subsets = subsets
        self._estimator_type = "classifier"
        self._n_outputs = 1
        self.subsets = []
        self.cols = {}
        self.verbose = verbose

        for ss in subsets:
            subset = tuple(df.columns.get_loc(col) for col in ss)
            self.subsets.append(subset)
            self.cols[subset] = ss

    def fit(self, X, y):
        # for each subset of features, train & test a decision tree
        self.trees = {}
        n_folds = 6
        self.scores = {ss: 0 for ss in self.subsets}

        for ss in self.subsets:
            # train each subset on a different set of folds
            folds = KFold(y.shape[0], n_folds=n_folds, shuffle=True)
            for train_index, test_index in folds:
                # make n folds of the data for training
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # generate a tree for each fold, and save the best one
                X_test_sub = X_test[:, ss]
                X_train_sub = X_train[:, ss]

                tree = sktree.DecisionTreeClassifier()
                tree.fit(X_train_sub, y_train)

                scorer = check_scoring(tree, scoring='roc_auc')
                score = (scorer(tree, X_test_sub, y_test) - 0.5) * 2

                if score > self.scores[ss]:
                    self.scores[ss] = score
                    self.trees[ss] = tree

        self.estimators_ = self.trees.values()

        if self.verbose:
            self.print_scores()

    def print_scores(self):
        for ss, score in sorted(self.scores.items(), key=lambda i: -i[1])[:3]:
            print "subset (%s): %.3f" % (', '.join(map(str, ss)), score)
            for pair in zip(ss, self.cols[ss]):
                print '\t%s: %s' % pair

    def predict_proba(self, X):
        """
        Take the average of the prob_a's of all trees
        """
        proba = 0
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset]) * self.scores[subset]

        proba /= sum(self.scores.values())
        return proba
