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


class SubsetForest(ForestClassifier):
    def __init__(self, df, labels, subsets=None, verbose=False,
                 tree_metric='f1', n_folds=5):
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
        self.tree_metric = tree_metric
        self.n_folds = n_folds

        for ss in subsets:
            subset = tuple(df.columns.get_loc(col) for col in ss)
            if subset not in self.subsets:
                self.subsets.append(subset)
                self.cols[subset] = ss

    def fit(self, X, y):
        # for each subset of features, train & test a decision tree
        self.trees = {}
        score_funcs = ['accuracy', 'roc_auc', 'f1']
        metrics = ['accuracy', 'roc_auc', 'f1', 'fp', 'fn']
        self.scores = pd.DataFrame(np.zeros((len(self.subsets), len(metrics))),
                                   index=map(str, self.subsets), columns=metrics)

        y = y.astype('bool')
        self.num_true = sum(y)
        self.num_false = sum(~y)

        for ss in self.subsets:
            # train each subset on a different set of folds
            folds = KFold(y.shape[0], n_folds=self.n_folds, shuffle=True)
            for train_index, test_index in folds:
                # make n folds of the data for training
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # generate a tree for each fold, and save the best one
                X_test_sub = X_test[:, ss]
                X_train_sub = X_train[:, ss]

                # create new decision tree classifier
                tree = sktree.DecisionTreeClassifier(class_weight='balanced')
                tree.fit(X_train_sub, y_train)

                # save some metrics about it
                scores = {}
                y_pred = tree.predict(X_test_sub)
                scores['fp'] = float(sum(y_pred & ~y_test)) / sum(~y_test)
                scores['fn'] = float(sum(~y_pred & y_test)) / sum(y_test)

                for met in score_funcs:
                    scorer = check_scoring(tree, scoring=met)
                    scores[met] = scorer(tree, X_test_sub, y_test)

                if scores[self.tree_metric] > self.scores[self.tree_metric][str(ss)]:
                    self.trees[ss] = tree
                    self.scores.loc[str(ss)] = pd.Series(scores)

                self.classes_ = tree.classes_

        self.estimators_ = self.trees.values()

        if self.verbose:
            self.print_scores()

    def print_scores(self):
        for ss, score in sorted(self.scores.iterrows(),
                                key= lambda i: -i[1][self.tree_metric])[:3]:
            print "subset %s: f1 = %.3f; roc_auc = %.3f; acc = %.3f" % \
                (ss, score['f1'], score['roc_auc'], score['accuracy'])

            subset = tuple([int(i) for i in ss[1:-1].split(',') if i != ''])
            for pair in zip(subset, self.cols[subset]):
                print '\t%s: %s' % pair

            #tree_to_code(self.trees[subset], self.cols[ss])

    def predict_proba_simple(self, X):
        """
        Take the average of the prob_a's of all trees
        """
        proba = np.zeros((len(X), 2))
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset]) * \
                self.scores[self.tree_metric][str(subset)]
        return proba

    def predict_proba(self, X):
        return self.predict_proba_simple(X)

        # Below here is commented out for now
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
