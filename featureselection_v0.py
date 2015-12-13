import copy
import itertools
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV


class FeatureSelection:
    
    def __init__(self, classifier, df):
        self.classifier = classifier
        self.X = df[ [col for col in df if col not in ['label', 'class']]]
        self.y = df['class'].values
        self.feature_labels = self.X.columns
        self.feature_types = self.X.dtypes.to_dict()
        
        # Binarize the categorical data using a DictVectorizer
        # This requires the data be fed in the form of Python dicts
        vectorizer = DV(sparse=False)
        X_binarized = vectorizer.fit_transform( self.X.to_dict(orient='records') )
        
        self.encoded_features = vectorizer.get_feature_names()

        # Split into test and train sets
        X_train, X_test, y_train, y_test = train_test_split(X_binarized, self.y)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.total_time = 0
        
        
    def feature_reduce(self):
        """
        """
        # Get the best number of features
        n = self._get_best_n()
        
        # Get all subsets of features of size n
        n_element_features = self._n_element_subsets(self.feature_labels, n)
        
        # Get best n features
        best_features, error = self._best_n_features(n_element_features, validation_set=True)
        
        print 'Total time taken: {}'.format(self.total_time)
        print 'Lowest Test error: {}'.format(error)
        
        return best_features, error
        
        
    def _n_element_subsets(self, S, n):
        """
        Return all of the n element subsets of S
        """
        return set(itertools.combinations(S, n))
    
        
    def _get_best_n(self):
        """
        """
        # Data for training and validating the model
        X = self.X_train
        y = self.y_train
        features = self.feature_labels
        
        # Save the features and the error for the best classifier
        # with n elements for each n
        best_n_model = []
        
        # Calculate error using all of the features
        classifier = copy.deepcopy(self.classifier)
        t0 = time()
        classifier.fit(X, y)
        error = (1 - classifier.score(X, y))
        t1 = time()
        self.total_time += (t1-t0)
        
        best_n_model.append( (features, error) )
        
        # Get the best n features for each n
        for n in range(self.X.shape[1]-1, 0, -1):
            # Get list of all n-element subsets of column labels
            n_feature_subsets = self._n_element_subsets(features, n)
            
            # Get best set of features of size n and lowest error
            best_n_features, best_n_error = self._best_n_features(n_feature_subsets)
            print 'Training error: {}'.format(best_n_error)
            best_n_model.append((best_n_features, best_n_error))
            
            # Reset feature to the best features from this model
            features = best_n_features
        
        best_n_model.sort(key=lambda x:x[1])
        print 'Total time taken: {}\n'.format(self.total_time)
    
        print 'Best n: {}'.format(len(best_n_model[0][0]))
        return len(best_n_model[0][0])
    
    
    def _best_n_features(self, feature_subsets, validation_set=False):
        """
        """
        print 'Calculating best {} features'.format(len(list(feature_subsets)[0]))
        X = self.X_train
        y = self.y_train
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
            
            # Get subset of X_train and X_test corresponding to the right features
            X_train_sub = X[:, feature_indices]
            
            # Fit and get error for classifier
            classifier = copy.deepcopy(self.classifier)
            t0 = time()
            classifier.fit(X_train_sub, y)
            
            if validation_set:
                X_test_sub = self.X_test[:, feature_indices]
                error = (1 - classifier.score(X_test_sub, self.y_test))
            else:
                error = (1 - classifier.score(X_train_sub, y))
                
            t1 = time()
            self.total_time += (t1-t0)
            
            n_feature_errors.append((feature_list, error))
            
        # Sort the feature lists and return the most performant feature list
        n_feature_errors.sort(key=lambda x:x[1])
        
        return n_feature_errors[0]