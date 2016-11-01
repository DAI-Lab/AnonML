import numpy as np
from random import shuffle
from sklearn import tree as sktree
from sklearn.ensemble.forest import ForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring

class SubsetForest(ForestClassifier):
    def __init__(self, df, labels, subsets=None, n_subsets=20, subset_size=4):
        self.labels = labels
        self.subsets = subsets
        self._estimator_type = "classifier"
        self._n_outputs = 1
        if subsets is None:
            self.subsets = []
            self.cols = {}

            # generate n_subsets subsets of subset_size features each
            print 'trying to generate %d random subsets...' % n_subsets
            shuf_cols = list(df.columns)
            shuffle(shuf_cols)

            for i in range(n_subsets):
                if not shuf_cols:
                    break
                cols = shuf_cols[:subset_size]
                shuf_cols = shuf_cols[subset_size:]
                subset = tuple(df.columns.get_loc(c) for c in cols)
                self.subsets.append(subset)
                self.cols[subset] = cols

            print 'generated %d non-overlapping subsets.' % len(self.subsets)

    def fit(self, X, y):
        # for each subset of features, make & train a decision tree
        self.trees = {}
        for subset in self.subsets:
            Xsub = X[:, subset]
            self.trees[subset] = sktree.DecisionTreeClassifier()
            self.trees[subset].fit(Xsub, y)

        self.estimators_ = self.trees.values()

        # for each tree, test & assign propensity score
        n_folds = 3
        self.scores = {ss: 0 for ss in self.trees}
        folds = KFold(y.shape[0], n_folds=n_folds, shuffle=True)

        for train_index, test_index in folds:
            # make 3 folds of the data for training
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # score each tree on this fold, and average the results across folds
            for ss, tree in self.trees.iteritems():
                scorer = check_scoring(tree, scoring='roc_auc')
                X_test_sub = X_test[:, ss]
                score = scorer(tree, X_test_sub, y_test)
                self.scores[ss] += score / n_folds

        for ss, score in sorted(self.scores.items(), key=lambda i: -i[1]):
            print "subset (%s): %.3f" % (', '.join(map(str, ss)), score)

        self.print_cols()

    def predict_proba(self, X):
        """
        Take the average of the prob_a's of all trees
        """
        proba = 0
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset])

        proba /= len(self.trees)
        return proba

    def print_cols(self):
        for ss, col in self.cols.items():
            print ss, col
