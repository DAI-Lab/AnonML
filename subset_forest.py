import numpy as np
import pdb
from IPython.core.debugger import Tracer
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
        self._estimator_type = 'classifier'
        self.n_outputs_ = 1
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
        self.scores = {ss: {'f1': 0, 'auc': 0} for ss in self.subsets}

        y = y.astype('bool')
        self.num_true = sum(y)
        self.num_false = sum(~y)

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

                tree = sktree.DecisionTreeClassifier(class_weight='balanced')
                tree.fit(X_train_sub, y_train)

                scorer = check_scoring(tree, scoring='f1')
                f1_score = scorer(tree, X_test_sub, y_test)
                scorer = check_scoring(tree, scoring='roc_auc')
                auc_score = scorer(tree, X_test_sub, y_test)

                y_pred = tree.predict(X_test_sub)
                false_pos = float(sum(y_pred & ~y_test)) / sum(~y_test)
                false_neg = float(sum(~y_pred & y_test)) / sum(y_test)

                if auc_score > self.scores[ss]['auc']:
                    self.trees[ss] = tree
                    self.scores[ss] = {'f1': f1_score,
                                       'auc': auc_score,
                                       'fp': false_pos,
                                       'fn': false_neg}

                self.classes_ = tree.classes_

        self.estimators_ = self.trees.values()

        if self.verbose:
            self.print_scores()

    def print_scores(self):
        for ss, score in sorted(self.scores.items(), key=lambda i: -i[1]['auc'])[:3]:
            print "subset (%s): f1 = %.3f; roc_auc = %.3f" % \
                (', '.join(map(str, ss)), score['f1'], score['auc'])
            for pair in zip(ss, self.cols[ss]):
                print '\t%s: %s' % pair

    def predict_proba_naive(self, X):
        """
        Take the average of the prob_a's of all trees
        """
        proba = np.zeros((len(X), 2))
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset]) * self.scores[subset]['auc']
        return proba

    def predict_proba(self, X):
        return self.predict_proba_naive(X)

        #best_subset = sorted(self.scores.items(), key=lambda i: -i[1]['f1'])[0][0]
        #return self.trees[best_subset].predict_proba(X[:, best_subset])

        lhs = np.zeros((len(X)))
        for subset, tree in self.trees.items():
            # True -> 1; False -> 0
            votes = tree.predict(X[:, subset]).astype('int')
            #pa = tree.predict_proba(X[:, subset])[:, 1]

            false_pos = self.scores[subset]['fp']
            false_neg = self.scores[subset]['fn']
            predictions = votes * np.log((1 - false_neg) / false_pos) + \
                (1 - votes) * np.log(false_neg / (1 - false_pos))

            #Tracer()()
            lhs += predictions

        lhs /= len(self.trees)
        rhs = np.log(float(self.num_true) / self.num_false)

        proba = np.ndarray((len(X), 2))
        for i, l in enumerate(lhs):
            # classes are [False, True]
            proba[i, :] = [0., 1.] if l > rhs else [1., 0.]

        return proba
