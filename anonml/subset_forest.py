import numpy as np
import pandas as pd
import sklearn
import pdb
from IPython.core.debugger import Tracer
from random import shuffle
from sklearn import tree as sktree
from sklearn.tree import _tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import ForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.scorer import check_scoring


# basically static variables
score_funcs = ['accuracy', 'roc_auc', 'f1']
metrics = score_funcs + ['fp', 'fn']


def print_tree_code(tree, feature_names):
    """
    Print out a decision tree as pseudocode
    """
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


def test_clf(clf, X_train, y_train, X_test, y_test):
    """ cross-validate a classifier """
    scores = {}

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # calculate false positive/false negative
    scores['fp'] = float(sum(y_pred & ~y_test)) / sum(~y_test) if \
        sum(~y_test) > 0 else 0
    scores['fn'] = float(sum(~y_pred & y_test)) / sum(y_test) if \
        sum(y_test) > 0 else 0

    # use all the scoring functions
    for met in score_funcs:
        scorer = check_scoring(clf, scoring=met)
        scores[met] = scorer(clf, X_test, y_test)

    return scores


class SubsetForest(ForestClassifier):
    """
    Ensemble classifier which combines classifiers on lots of vertical
    partitions into a single model
    """
    def __init__(self, clf_metric='f1', n_folds=3, cols=None,
                 max_tree_depth=None, verbose=False):
        """
        clf_metric (str): One of ('f1', 'roc_auc', 'accuracy'). Determines which score
            is used to weight the subset classifiers
        n_folds (int): number of folds on which to test each classifier
        cols (list[str]): column names
        verbose (bool): whether or not to print extra information
        """
        self._estimator_type = 'classifier'
        self.n_outputs_ = 1
        self.cols = cols
        self.max_tree_depth = max_tree_depth
        self.verbose = verbose
        self.clf_metric = clf_metric
        self.n_folds = n_folds

    def fit(self, training_data):
        """
        training_data: dict mapping subsets to (X, y) matrix-array tuples

        For each subset of features, train and test a classifier
        """
        if self.verbose:
            print
            print 'Fitting Subset Forest with %d subsets' % len(training_data)

        # dictionary mapping a classifier (identified by a feature subset) to
        # its set of performance scores
        self.scores = {subset: {met: 0 for met in metrics}
                       for subset in training_data}
        trials = {subset: 0 for subset in training_data}

        # classifiers keyed by subsets
        self.classifiers = {}

        # stats
        self.num_true = 0
        self.num_false = 0

        if self.verbose:
            print "\ttesting subset classifiers..."

        # generate a classifier for each subset, and test it on several folds of data
        for subset, (X, y) in training_data.items():
            if self.verbose:
                print "\ttesting classifier", subset, "on", len(X), "samples"

            # count stats about the labels
            y = y.astype('bool')
            self.num_true += sum(y)
            self.num_false += sum(~y)

            # if this set has only one label, don't make a classifier
            if sum(y) == 0 or sum(~y) == 0:
                print "Can't train on %s: all labels are the same" % str(subset)
                del training_data[subset]
                continue

            # make k folds of the data for training
            folds = KFold(n_splits=self.n_folds, shuffle=True).split(X)

            n_scores = 0
            for i, (train_index, test_index) in enumerate(folds):
                if self.verbose:
                    print "\t\tfold %d/%d" % (i+1, self.n_folds)

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # create new decision tree classifier
                clf = sktree.DecisionTreeClassifier(class_weight='balanced',
                                                     max_depth=self.max_tree_depth)
                # Actually, let's make it a regression
                #clf = LogisticRegression(class_weight='balanced')

                # sometimes this doesn't work because of
                try:
                    scores = test_clf(clf, X_train, y_train, X_test, y_test)
                except Exception as e:
                    print e
                    continue

                # save average metrics for each subset
                for k in scores:
                    self.scores[subset][k] += float(scores[k])
                trials[subset] += 1

        if self.verbose:
            print "\ttraining subset classifiers"

        # normalize scores
        for subset in training_data.keys():
            for k in self.scores[subset]:
                self.scores[subset][k] /= max(trials[subset], 1)

        for subset, (X, y) in training_data.items():
            # train classifier on whole dataset
            clf = sktree.DecisionTreeClassifier(class_weight='balanced',
                                                 max_depth=self.max_tree_depth)
            #clf = LogisticRegression(class_weight='balanced')
            clf.fit(X, y)
            self.classifiers[subset] = clf
            self.classes_ = clf.classes_

        self.estimators_ = self.classifiers.values()

        if self.verbose:
            print "Fit complete."
            print

    def predict_proba(self, X):
        return self.predict_proba_simple(X)

    def predict_proba_vote(self, X):
        """
        Each clf gets a binary vote. Each clf's vote is weighted by its score,
        then they are summed.
        """
        proba = np.zeros((len(X), 2))
        for subset, clf in self.classifiers.items():
            for i, p in enumerate(clf.predict(X[:,subset])):
                # each clf gets a vote, T or F.
                # p is a label, True or False
                # each clf's vote is weighted by that clf's score.
                # the particular scoring metric we're using is self.clf_metric
                # classes are [False, True]
                vote = self.scores[subset][self.clf_metric]
                prob = [0., vote] if p else [vote, 0.]
                proba[i, :] += prob
        return proba

    def predict_proba_simple(self, X):
        """
        Take the average of the prob_a's of all clfs, weighted by
        self.clf_metric.
        """
        proba = np.zeros((len(X), 2))
        for subset, clf in self.classifiers.items():
            proba += clf.predict_proba(X[:, subset]) * \
                self.scores[subset][self.clf_metric]
        return proba

    def predict_proba_complicated(self, X):
        """
        From Kalyan's thesis. Not used for now.
        """
        lhs = np.zeros((len(X)))
        for subset, clf in self.classifiers.items():
            # True -> 1; False -> 0
            votes = clf.predict(X[:, subset]).astype('int')
            #pa = clf.predict_proba(X[:, subset])[:, 1]

            false_pos = self.scores[subset]['fp']
            false_neg = self.scores[subset]['fn']
            predictions = votes * np.log((1 - false_neg) / false_pos) + \
                (1 - votes) * np.log(false_neg / (1 - false_pos))

            lhs += predictions

        lhs /= len(self.classifiers)
        rhs = np.log(float(self.num_true) / self.num_false)

        proba = np.ndarray((len(X), 2))
        for i, l in enumerate(lhs):
            # classes are [False, True]
            proba[i, :] = [0., 1.] if l > rhs else [1., 0.]

        return proba

    def print_scores(self):
        for ss, score in sorted(self.scores.items(),
                                key=lambda i: -i[1][self.clf_metric])[:3]:
            print "subset %s: f1 = %.3f; roc_auc = %.3f; acc = %.3f" % \
                (ss, score['f1'], score['roc_auc'], score['accuracy'])

            if len(self.cols):
                for ix in ss:
                    print '\t%s: %s' % (ix, self.cols[ix])

            #print_tree_code(self.classifiers[ss], self.cols[ss])
