from scipy.special import expit as sigmoid
import numpy as np


class LogisticRegression:
    """
    Implementation of Logistic Regression classifier with options for
    L1 and L2 regularization.
    
    author: Luigi Patruno    
    """
    
    def __init__(self):
        """
        Initialize a Logistic Regression Classifier object
        
        Params
        ---------
        weights - Learned weights for the Logistic Regression.
            Fit to the training data by calling the fit method.
            
        _epsilon - Hyperparameter for Logistic Regression controlling overfitting.
            
        _num_training - Number of training data points. Used for regularization.
        
        _lambda - Hyperparameter controlling regularization.
        """
        self.weights = None
        self._epsilon = None
        self._num_training = None
        self._lambda = None
        return None
    
    
    def accuracy(self, y, x):
        """
        Returns the classification accuracy.
        """
        prediction = self.predict(x)
        return np.mean(prediction == y)
    
    
    def fit(self, y, x, n=1, epsilon=.01, regularization=None, _lambda=1.0):
        """
        Learn the weights for a Logistic Regression Classifier from the data
        
        Params
        ---------
        y : Numpy array
            - The binary class labels for the data points
        x : Numpy array
            - A list of features for each data point
        n : int - Default: 1
            - The number of iterations for the training algorithm
        epsilon : float - Default: 0.01
            - Hyperparamter controlling overfitting of the model to the data
        regularization : {None, 'L1', 'L2'} - Default: None
            - Type of regularization to employ on the model
        _lambda : float
            - Hyper parameter for regularization
        """
        # Initialize the weight vector
        w_0 = np.zeros(x.shape[1])
        
        # Variables used for learning weights
        self._epsilon = epsilon
        self._num_training = x.shape[0]
        self._lambda = _lambda
        
        print 'Epsilon: {}'.format(self._epsilon)
        
        # Pick the correct update method
        if regularization == 'l1':
            print 'L1 regularization'
            print 'Lambda: {}'.format(self._lambda)
            update_func = self._l1
        elif regularization == 'l2':
            print 'L2 regularization'
            print 'Lambda: {}'.format(self._lambda)
            update_func = self._l2
        else:
            print 'No regularization'
            update_func = self._no_reg
                    
        # Number of iterations
        for _ in range(n):

            # Loop over all the data points
            for i in range(x.shape[0]):
                
                y_minus_g = y[i] - sigmoid( np.dot(w_0, x[i]) )
                w_0 = update_func(y[i], x[i], y_minus_g, w_0)

        # Save the learned weights
        self.weights = w_0
        return None
    
    
    def _no_reg(self, y, x, y_minus_g, w_0):
        """
        Calculate the update to the weight vector with no regularization.
        
        Params
        ---------
        y : {0, 1}
            The binary class label for the data point
        x : Numpy array
            The feature vector for a single data point
        y_minus_g : float
            (y - sigmoid(w^T x))
        w_0 : float
            Previous weight vector
        epsilon : float
            Hyperparameter controlling overfitting
            
        Returns 
        ---------
            - New weight vector
        """
        # Initialize weight vector to return
        w_1 = np.zeros(len(w_0))
        
        ascent = self._epsilon * y_minus_g
        
        for j in range(len(x)):
            w_1[j] = w_0[j] + x[j]*ascent
            
        return w_1
    
    
    def _l1(self, y, x, y_minus_g, w_0):
        """
        Update rule for Logistic Regression with L1 regularization.
        
        Params
        ---------
        y : {0, 1}
            The binary class label for the data point
        x : Numpy array
            The feature vector for a single data point
        y_minus_g : float
            (y - sigmoid(w^T x))
        w_0 : float
            Previous weight vector
        epsilon : float
            Hyperparameter controlling overfitting
            
        Returns 
        ---------
            - New weight vector
        """
        # Initialize weight vector to return
        w_1 = np.zeros(len(w_0))
        
        for j in range(len(x)):
            reg = float(self._sign(w_0[j])) / (self._lambda*self._num_training)
            w_1[j] = w_0[j] + self._epsilon*(x[j]*y_minus_g - reg)
            
        return w_1
    
    
    def _l2(self, y, x, y_minus_g, w_0):
        """
        Update rule for Logistic Regression with L2 regularization.
        
        Params
        ---------
        y : {0, 1}
            The binary class label for the data point
        x : Numpy array
            The feature vector for a single data point
        y_minus_g : float
            (y - sigmoid(w^T x))
        w_0 : float
            Previous weight vector
        epsilon : float
            Hyperparameter controlling overfitting
            
        Returns 
        ---------
            - New weight vector
        """
        # Initialize weight vector to return
        w_1 = np.zeros(len(w_0))
        
        for j in range(len(x)):
            reg = float(w_0[j]) / (self._lambda*self._num_training)
            w_1[j] = w_0[j] + self._epsilon*(x[j]*y_minus_g - reg)
            
        return w_1
    
    
    def _sign(self, number): 
        """
        Returns the sign of a number (0 if number == 0)
        """
        return cmp(number,0)
    
    
    def predict(self, data):
        """
        Classifies each data point in x according to the weights learned from the fit method.

        Params
        ---------
        x : Numpy array
            The array of data points to classify.
            
        Returns
        ---------
            - Predicted class label for each data point.
        """
        prediction = []

        for x in data:
            prob_0 = self._sigmoidLikelihood(x, 0)
            prob_1 = self._sigmoidLikelihood(x, 1)

            if prob_0 > prob_1:
                prediction.append(0)
            else:
                prediction.append(1)

        return prediction
    
    
    def _sigmoidLikelihood(self, x, label):
        """
        Returns the sigmoid likelihood p(y=label|features; weights)
        
        x : Numpy array
            Feature set for a single data point
        """
        logit = sigmoid(np.dot(x, self.weights))
        
        if label == 0:
            return (1-logit)
        elif label == 1:
            return logit