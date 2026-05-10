"""minilearn classifiers package."""
from minilearn.classifiers.knn import KNeighborsClassifier
from minilearn.classifiers.naive_bayes import GaussianNB
from minilearn.classifiers.logistic_regression import LogisticRegression
from minilearn.classifiers.decision_tree import DecisionTreeClassifier
from minilearn.classifiers.mlp import MLPClassifier
from minilearn.classifiers.svm import LinearSVM

__all__ = ["KNeighborsClassifier", "GaussianNB",
           "LogisticRegression", "DecisionTreeClassifier",
           "MLPClassifier", "LinearSVM"]