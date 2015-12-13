"""
Driver to run the Logistic Regression with L1 regularization analysis.

author: Luigi Patruno
"""

from time import time

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.linear_model import LogisticRegression

from get_data import data

def main():
    df = data()
    
    X = df[ [col for col in df if col not in ['class']]]
    y = df['class'].values

    # Binarize the categorical data using a DictVectorizer
    # This requires the data be fed in the form of Python dicts
    vectorizer = DV(sparse=False)
    X_binarized = vectorizer.fit_transform( X.to_dict(orient='records') )


    # Split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, train_size=.8, random_state=42)
    
    C = np.linspace(.1, 10, 100).tolist() + np.linspace(20,100, 5).tolist() + \
        np.linspace(200, 1000, 9).tolist()
        
    param_grid = list(ParameterGrid({'C': C, 'penalty': ['l1']}))
    
    for params in param_grid:
    
        classifier = LogisticRegression(**params)

        # Fit the model to the training data
        t0 = time()
        classifier.fit(X_train, y_train)
        t1 = time()

        accuracy = classifier.score(X_test, y_test)
        error = (1 - accuracy)

        print '\nTest error: {} Time to train: {}'.format(error, (t1-t0))
        print 'Params: {}'.format(params)
    
    
if __name__ == '__main__':
    main()