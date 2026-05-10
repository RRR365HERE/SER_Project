"""minilearn classifiers package."""
from minilearn.classifiers.knn import KNeighborsClassifier
from minilearn.classifiers.naive_bayes import GaussianNB
from minilearn.classifiers.logistic_regression import LogisticRegression
from minilearn.classifiers.decision_tree import DecisionTreeClassifier

__all__ = ["KNeighborsClassifier", "GaussianNB",
           "LogisticRegression", "DecisionTreeClassifier"]