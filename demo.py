import pandas as pd
import numpy as np
import pdb
from sklearn import tree as sktree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble.forest import ForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics.scorer import check_scoring


def test_random_forest(X, y):
    X = np.nan_to_num(X)
    y = np.array(y)
    folds = KFold(y.shape[0], n_folds=3, shuffle=True)
    results = []
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        scorer = check_scoring(clf, scoring='roc_auc')
        results.append(scorer(clf, X_test, y_test))

    npres = np.array(results)
    print 'Random forest: mean = %f, std = %f' % (npres.mean(), npres.std())


def test_random_subspaces(X, y):
    X = np.nan_to_num(X)
    y = np.array(y)
    folds = KFold(y.shape[0], n_folds=3, shuffle=True)
    results = []
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = BaggingClassifier(sktree.DecisionTreeClassifier(), max_samples=0.5,
                                max_features=2)
        clf.fit(X_train, y_train)
        scorer = check_scoring(clf, scoring='roc_auc')
        results.append(scorer(clf, X_test, y_test))

    npres = np.array(results)
    print 'Random subspaces: mean = %f, std = %f' % (npres.mean(), npres.std())


def test_random_small_forests(df, labels, subsets=None, n_subsets=25, subset_size=4):
    classifier = SubsetForest(df, labels, subsets, n_subsets, subset_size)
    X = np.nan_to_num(df.as_matrix())
    pdb.set_trace()
    y = labels
    folds = KFold(y.shape[0], n_folds=3, shuffle=True)
    for train_index, test_index in folds:
        # make 3 folds of the data for training
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)

        # score the superclassifier
        scorer = check_scoring(classifier, scoring='roc_auc')
        print 'Random subspace forest: Total score =', scorer(classifier, X_test, y_test)



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
            for i in range(n_subsets):
                cols = np.random.choice(df.columns, subset_size)
                subset = tuple(df.columns.get_loc(c) for c in cols)
                if self.validate_subset(subset):
                    self.subsets.append(subset)
                    self.cols[subset] = cols

    def validate_subset(self, subset):
        """
        Returns True if the subset of features meats certain criteria for
        validity - in this case, at most one overlapping feature with any other
        subset.
        """
        for ss in self.subsets:
            # if the new subset has more than one feature in common with any
            # other subset, return False
            if len(set(subset) & set(ss)) > 1:
                return False
        return True

    def fit(self, X, y):
        # for each subset of features, make & train a decision tree
        self.trees = {}
        for subset in self.subsets:
            Xsub = X[:, subset]
            self.trees[subset] = sktree.DecisionTreeClassifier()
            self.trees[subset].fit(Xsub, y)

        self.estimators_ = self.trees.values()

        # for each tree, test & assign propensity score
        #self.scores = {}
        #folds = KFold(y.shape[0], n_folds=3, shuffle=True)
        #for train_index, test_index in folds:
            ## make 3 folds of the data for training
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]

            #for ss, tree in self.trees.iteritems():
                #scorer = check_scoring(tree, scoring='roc_auc')
                #X_test_sub = X_test[:, ss]
                #print ss, scorer(tree, X_test_sub, y_test)

    def predict_proba(self, X):
        """
        Take the average of the prob_a's of all trees
        """
        proba = 0
        for subset, tree in self.trees.items():
            proba += tree.predict_proba(X[:, subset])

        proba /= len(self.trees)
        return proba

def main():
    df = pd.read_csv(open('./3091x_f12_combined.csv'))
    labels = df['dropout'].values
    del df['dropout']

    #test_random_subspaces(df.as_matrix(), labels)
    #test_random_forest(df.as_matrix(), labels)

    test_random_small_forests(df, labels)

if __name__ == '__main__':
    main()
