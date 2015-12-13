import copy
import itertools
from time import time

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import LogisticRegression

from get_data import data


class FeatureSelection:
    
    def __init__(self, df, classifiers):
        self.total_time = 0
        self.df = df
        self.classifiers = classifiers
        # Split and save train, validation and test splits
        self._split_data(train_size=.6, test_size=.4)
        
    def select_features(self):
        """
        Perform a feature selection over a list of classifiers
        with certain hyperparameters set.
        """
        results = []
        # Loop over all the classifiers and hyperparameters
        for model in self.classifiers:
                
            classifier_name = model['classifier_name']
            param_grid = list(ParameterGrid(model['param_grid']))
            

            for params in param_grid:
                # Initialize the model
                classifier = copy.deepcopy( model['classifier'] )
                # Initialize a model with the hyperparameters
                classifier = classifier(**params)
                print('\n\n{}'.format(params))
                # Get subset of features that minimizes the validation error 
                features, error = self._feature_reduce(classifier)
                # Save results
                classifier_result = {'classifier': classifier_name, 'params':str(params), 'features': '{}'.format(features), 'error':error}
                results.append(classifier_result)
                
        return results
    
        
    def _split_data(self, train_size=.8, test_size=.2):
        """
        Encode the categorical variables using a One-Hot encoding
        and split the data into training, validation and test splits.
        
        Parameters
        ----------
        train_size : float
            Pct of data to use as training samples
        test_size : float
            Pct of data to use as validation and test samples
        """
        # Get features and target
        X = self.df[ [col for col in self.df if col not in ['label', 'class']]]
        y = self.df['class'].values
        
        # Binarize the categorical data using a DictVectorizer
        # This requires the data be fed in the form of Python dicts
        vectorizer = DV(sparse=False)
        print('Encoding features...')
        X_binarized = vectorizer.fit_transform(X.to_dict(orient='records'))
        
        # Split into train, cv and test sets
        X_train, X_cv_test, y_train, y_cv_cv = train_test_split(X_binarized, y, 
                                                            train_size=train_size, test_size=test_size)
        X_cv, X_test, y_cv, y_test = train_test_split(X_cv_test, y_cv_cv, 
                                                            train_size=(test_size/2), test_size=(test_size/2))
        
        self.encoded_features = vectorizer.get_feature_names()
        self.feature_labels = X.columns
        self.feature_types = X.dtypes.to_dict()
        self.X = X
        self.X_train = X_train
        self.X_cv = X_cv
        self.X_test = X_test
        self.y_train = y_train
        self.y_cv = y_cv
        self.y_test = y_test
        
        print('Number of training samples: {}'.format(X_train.shape[0]))
        print('Number of validation samples: {}'.format(X_cv.shape[0]))
        print('Number of test samples: {}'.format(X_test.shape[0]))
    
        
    def _feature_reduce(self, classifier):
        """
        """
        # Get the best number of features
        n = self._get_best_n(classifier)
        
        # Get all subsets of features of size n
        n_element_features = self._n_element_subsets(self.feature_labels, n)
        
        # Get best n features
        best_features, error = self._best_n_features(classifier, n_element_features, test_set=True)
        
        print 'Total time taken: {}'.format(self.total_time)
        print 'Lowest Test error: {}'.format(error)
        
        return best_features, error
        
        
    def _n_element_subsets(self, S, n):
        """
        Return all of the n element subsets of S
        """
        return set(itertools.combinations(S, n))
    
        
    def _get_best_n(self, classifier):
        """
        """
        # Data for training and validating the model
        X_train = self.X_train
        y_train = self.y_train
        X_cv = self.X_cv
        y_cv = self.y_cv
        
        features = self.feature_labels
        
        # Save the features and the error for the best classifier
        # with n elements for each n
        best_n_model = []
        
        # Calculate error using all of the features
        classifier = copy.deepcopy(classifier)
        t0 = time()
        classifier.fit(X_train, y_train)
        error = (1 - classifier.score(X_cv, y_cv))
        t1 = time()
        self.total_time += (t1-t0)
        
        print 'Training error for {} features: {}'.format(self.X.shape[1], error)
        
        best_n_model.append( (features, error) )
        
        # Get the best n features for each n
        for n in range(self.X.shape[1]-1, 0, -1):
            # Get list of all n-element subsets of column labels
            n_feature_subsets = self._n_element_subsets(features, n)
            
            # Get best set of features of size n and lowest error
            best_n_features, best_n_error = self._best_n_features(classifier, n_feature_subsets)
            print 'Training error: {}'.format(best_n_error)
            best_n_model.append((best_n_features, best_n_error))
            
            # Reset feature to the best features from this model
            features = best_n_features
        
        best_n_model.sort(key=lambda x:x[1])
        print 'Total time taken: {}\n'.format(self.total_time)
    
        print 'Best n: {}'.format(len(best_n_model[0][0]))
        return len(best_n_model[0][0])
    
    
    def _best_n_features(self, classifier, feature_subsets, test_set=False):
        """
        """
        print 'Calculating best {} features'.format(len(list(feature_subsets)[0]))
        
        enc_feats = self.encoded_features
        
        # Keep track of the error for each of the n feature classifiers
        n_feature_errors = []
        
        # Loop through each list of n-element column features
        for feature_list in feature_subsets:
            
            # Get list of indices of the features
            # Non-trivial due to the OneHotEncoder-ing of the data
            feature_indices = []
            
            # Loop over each feature within the list to get types
            for feature in feature_list:
                if self.feature_types[feature] == object:
                    encoded_feature_label = feature + '='
                    encoded_feat_indices = [enc_feats.index(col) for col in enc_feats if encoded_feature_label in col]
                    feature_indices += encoded_feat_indices
                else:
                    feature_indices.append(enc_feats.index(feature))
            
            # Fit and get error for classifier
            classifier = copy.deepcopy(classifier)
            
            if test_set:
                # Use training and cv sets to train the model
                X_train = np.concatenate((self.X_train,self.X_cv))
                y_train = np.concatenate((self.y_train, self.y_cv))
                # Get subset of X_train and X_test corresponding to the right features
                X_train = X_train[:, feature_indices]
                
                t0 = time()
                classifier.fit(X_train, y_train)
                
                X_test = self.X_test[:, feature_indices]
                error = (1 - classifier.score(X_test, self.y_test))
            else:
                # Get subset of X_train and X_test corresponding to the right features
                X_train = self.X_train[:, feature_indices]
                y_train = self.y_train
        
                t0 = time()
                classifier.fit(X_train, y_train)
                X_cv = self.X_cv[:, feature_indices]
                error = (1 - classifier.score(X_cv, self.y_cv))
                
            t1 = time()
            self.total_time += (t1-t0)
            
            n_feature_errors.append((feature_list, error))
            
        # Sort the feature lists and return the most performant feature list
        n_feature_errors.sort(key=lambda x:x[1])
        
        return n_feature_errors[0]