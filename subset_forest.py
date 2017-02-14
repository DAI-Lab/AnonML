import numpy as np
import pandas as pd
import pdb
from IPython.core.debugger import Tracer
from random import shuffle
from sklearn import tree as sktree
from sklearn.tree import _tree
from sklearn.ensemble.forest import ForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)


class SubspaceForest(ForestClassifier):
    def __init__(self, verbose=False, tree_metric='f1', n_folds=3, cols=None):
        self._estimator_type = 'classifier'
        self.n_outputs_ = 1
        self.cols = cols
        self.verbose = verbose
        self.tree_metric = tree_metric
        self.n_folds = n_folds

    def fit(self, training_data):
        """
        training_data: dict mapping subsets to (X, y) matrix-array tuples

        For each subset of features, train and test a decision tree
        """
        if self.verbose:
            print
            print 'Fitting Subspace Forest with %d subsets' % len(training_data)

        # basically static variables
        score_funcs = ['accuracy', 'roc_auc', 'f1']
        metrics = score_funcs + ['fp', 'fn']

        # dictionary mapping a decision tree (identified by a feature subset) to
        # its set of performance scores
        self.scores = {subset: {met: 0 for met in metrics} for subset in training_data}

        # decision trees keyed by subsets
        self.trees = {}

        # stats
        self.num_true = 0
        self.num_false = 0

        if self.verbose:
            print "\ttesting subset trees..."

        # generate a tree for each subset, and test it on several folds of data
        for subset, (X, y) in training_data.iteritems():
            if self.verbose:
                print "\ttesting tree", subset

            # count stats about the labels
            y = y.astype('bool')
            self.num_true += sum(y)
            self.num_false += sum(~y)

            # make k folds of the data for training
            folds = KFold(y.shape[0], n_folds=self.n_folds, shuffle=True)

            for i, (train_index, test_index) in enumerate(folds):
                if self.verbose:
                    print "\t\tfold %d/%d" % (i+1, self.n_folds)

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # create new decision tree classifier
                tree = sktree.DecisionTreeClassifier(class_weight='balanced')
                tree.fit(X_train, y_train)

                # cross-validate this tree
                scores = {}
                y_pred = tree.predict(X_test)

                # calculate false positive/false negative
                scores['fp'] = float(sum(y_pred & ~y_test)) / sum(~y_test)
                scores['fn'] = float(sum(~y_pred & y_test)) / sum(y_test)

                # use all the scoring functions
                for met in score_funcs:
                    scorer = check_scoring(tree, scoring=met)
                    scores[met] = scorer(tree, X_test, y_test)

                # save average metrics for each tree
                for k in scores:
                    self.scores[subset][k] += float(scores[k]) / self.n_folds

        if self.verbose:
            print "\ttraining subset trees"

        for subset, (X, y) in training_data.iteritems():
            # train classifier on whole dataset
            tree = sktree.DecisionTreeClassifier(class_weight='balanced')
            tree.fit(X, y)
            self.trees[subset] = tree
            self.classes_ = tree.classes_

        self.estimators_ = self.trees.values()

        if self.verbose:
            print "Fit complete."
            print

    def predict_proba(self, X):
        return self.predict_proba_simple(X)

    def predict_proba_vote(self, X):
        """
        Take the average of the prob_a's of all trees, weighted by
        """
        proba = np.zeros((len(X), 2))
        for subset, tree in self.trees.items():
            for i, p in enumerate(tree.predict(X[:,subset])):
                # classes are [False, True]
                vote = self.scores[subset][self.tree_metric]
                prob = [0., vote] if p else [vote, 0.]
                proba[i, :] += prob
        return proba

    def predict_proba_simple(self, X):
        """
        Take the average of the prob_a's of all trees, weighted by <some metric>
        """
        proba = np.zeros((len(X), 2))
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset]) * \
                self.scores[subset][self.tree_metric]
        return proba

    def predict_proba_complicated(self, X):
        """
        From Kalyan's thesis. Not used for now.
        """
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

    def print_scores(self):
        for ss, score in sorted(self.scores.items(),
                                key=lambda i: -i[1][self.tree_metric])[:3]:
            print "subset %s: f1 = %.3f; roc_auc = %.3f; acc = %.3f" % \
                (ss, score['f1'], score['roc_auc'], score['accuracy'])

            if self.cols:
                for pair in zip(ss, self.cols[ss]):
                    print '\t%s: %s' % pair

            #tree_to_code(self.trees[ss], self.cols[ss])
