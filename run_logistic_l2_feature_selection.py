"""
Driver to run the Logistic Regression with L2 regularization analysis.
This includes performing feature reduction.

Due to the amount of time it takes to run the feature reduction with the validation set approach, 
I've included a test run that can be performed by running this script.
To run the full analysis, uncomment the commented block of code on lines 37-39.

author: Luigi Patruno
"""
import copy
import itertools
from time import time

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import LogisticRegression

from get_data import data
from featureselection import FeatureSelection


def main():
    df = data()

    # Test run
    classifiers = [{'classifier': LogisticRegression, 
                   'classifier_name': 'Logistic Regression L2',
                    'param_grid': {'C': [.1], 'max_iter': [10], 'penalty': ['l2']} }]

    # To complete the full analysis, please uncomment the following block of code
    # and run this script. 
    # NOTE This will take several hours to run
    
    ##classifiers = [{'classifier': LogisticRegression, 
    ##               'classifier_name': 'Logistic Regression L2',
    ##                'param_grid': {'C': np.linspace(.1, 5, 50), 'max_iter': [10,50,100,200], 'penalty': ['l2']} }]

    feature_selection = FeatureSelection(df, classifiers)

    feature_selection.select_features()
    
if __name__ == '__main__':
    main()