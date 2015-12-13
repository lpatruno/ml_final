"""
Driver to run a test example for my implementation of Logistic Regression with L1 
and L2 regularization.

I did not use these numbers in my official analysis because I was unhappy with the
performance of my implementation compared to the scikit-learn class with respect
to speed and accuracy. However, I have included a test driver here to show that
the model runs. Hey, I spent the couple of hours programming that routine, might
as well.

author: Luigi Patruno
"""
from time import time

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV

from get_data import data
from logisticregression import LogisticRegression

def main():
    # Load the data set
    df = data()

    X = df[ [col for col in df if col not in ['label', 'class']]]
    y = df['class'].values

    # Binarize the categorical data using a DictVectorizer
    # This requires the data be fed in the form of Python dicts
    vectorizer = DV(sparse=False)
    X_binarized = vectorizer.fit_transform(X.to_dict(orient='records'))

    X_binarized = np.array(X_binarized)

    # Split into train, cv and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, train_size=.8, random_state=42)
    
    # My implementation
    classifier = LogisticRegression()
    classifier.fit(y_train, X_train, regularization='l1')
    l1_error = 1 - classifier.accuracy(y_test, X_test)
    
    classifier = LogisticRegression()
    classifier.fit(y_train, X_train, regularization='l2')
    l2_error = 1 - classifier.accuracy(y_test, X_test)
    
    
    print('LogisticRegression with L1 regularization \nError: {}'.format(l1_error))
    print('LogisticRegression with L2 regularization \nError: {}'.format(l2_error))
    
    
if __name__ == '__main__':
    main()