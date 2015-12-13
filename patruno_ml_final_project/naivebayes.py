import numpy as np

class NaiveBayes:
    
    def __init__(self, ignore_missing=False):
        """
        Initialize a NaiveBayes object.
            
        Parameters
        ----------
        ignore_missing : boolean (Default: False)
            Whether to ignore feature that contain missing values
            
        Attributes
        ----------
        learned_params : dict
            Parameters learned from the Naive Bayes model
        features : list(str)
            Labels of all features to use
        categorical_features : list(str)
            Labels of features that are categorical
        """
        self.ignore_missing = ignore_missing
        
        if self.ignore_missing:
            self.features = ['age', 'fnlwgt', 'education', 'education-num', 'marital-status', \
                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
                            'hours-per-week']
            self.categorical_features = ['education', 'marital-status', 'relationship', 'race', 'sex']
            
        else:
            self.features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
                            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
                            'hours-per-week', 'native-country']
            self.categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country']
            
        self.learned_params = None
        
            
    def gaussian_pdf(self, mu, sigma, x):
        """
        Returns the probability of a point x according to a Gaussian distribution
        with mean mu and standard deviation sigma.
        """
        return 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu)**2 / (2 * sigma**2) )
    
    
    def score(self, samples):
        """
        Return the classification accuracy for a given set of samples.
        
        Parameters
        ----------
        samples : pandas DataFrame
            - Samples of data to predict
            
        Returns
        ----------
            - (Float) The classification accuracy of the Bayes model on the samples
        """
        y_pred = self.classify(samples.to_dict(orient='records'))
        y_true = samples['class'].tolist()
        
        return np.mean( np.array(y_true) == np.array(y_pred) )
        
            
    def classify(self, samples):
        """
        Classifies new data points according to the Naive Bayes model learned from
        the learn_parameters function. 
        
        Parameters
        ----------
        samples : pandas DataFrame
            The samples of data to classify
            
        Returns
        ----------
            - (List) The predicted class labeles for each of the samples.
        """
        
        if self.learned_params is None:
            print('Please call learn_parameters to fit the Naive Bayes model.')
            return
            
        predictions = []
        
        categorical_features = self.categorical_features
        learned_params = self.learned_params
        
        for sample in samples:
            
            class_0 = []
            class_1 = []
            
            for feature in sample:
                if feature in learned_params:
                    
                    value = sample[feature]
                    
                    if feature in categorical_features:
                        class_0_prob = learned_params[feature][value][0]
                        class_1_prob = learned_params[feature][value][1]
                        class_0.append(class_0_prob)
                        class_1.append(class_1_prob)
                    else:
                        c0_mean = learned_params[feature][0]['mean']
                        c0_std = learned_params[feature][0]['std']
                        c1_mean = learned_params[feature][1]['mean']
                        c1_std = learned_params[feature][1]['std']
                        class_0_prob = self.gaussian_pdf(c0_mean, c0_std, value)
                        class_1_prob = self.gaussian_pdf(c1_mean, c1_std, value)
                        class_0.append(class_0_prob)
                        class_1.append(class_1_prob)
                        
            c_0_prob = np.prod(class_0)
            c_1_prob = np.prod(class_1)
                    
            if c_0_prob > c_1_prob:
                predictions.append(0)
            else:
                predictions.append(1)
                        
        return predictions
        
            
    def learn_parameters(self, training_set):
        """
        Learn the parameters for the Naive Bayes model according to the data.
        
        If the feature is categorical, the parameter learned are the fraction
        of the samples containing a particular value for that feature, per class.
        
        If the feature is numerical, the parameters learned are the mean and 
        standard deviation of the feature, per class.
        
        These parameters are stored in the object atribute `learned_params`.
        
        Parameters
        ----------
        training_set : pandas DataFrame
            The data from which to learn the Naive Bayes model
        """
        df = training_set
        
        features = self.features
        categorical_features = self.categorical_features

        num_class_0 = df[df['class'] == 0].shape[0]
        num_class_1 = df[df['class'] == 1].shape[0]

        # Hold the learned parameters
        # Mean and std for continuous quantities
        # Feature-value probabilities for categorical data
        learned_probabilities = {}

        for feature in features:

            # Learn probabilities for categorical variables
            if feature in categorical_features:  
                # List of unique values the feature can take
                unique_values = df[feature].unique()
                # Probabilities for given values conditioned on class
                conditional_probs = {}

                # Calculate these probabilities
                for value in unique_values:
                    class_0_prob = df[ (df[feature] == value) & (df['class'] == 0) ].shape[0] / float(num_class_0)
                    class_1_prob = df[ (df[feature] == value) & (df['class'] == 1) ].shape[0] / float(num_class_1)
                    conditional_probs[value] = {0 : class_0_prob, 1 : class_1_prob}

                learned_probabilities[feature] = conditional_probs

            else:
                # Get the mean and std for each class
                class_0_mean = df[ df['class'] == 0 ][feature].mean()
                class_0_std = df[ df['class'] == 0 ][feature].std()
                class_1_mean = df[ df['class'] == 1 ][feature].mean()
                class_1_std = df[ df['class'] == 1 ][feature].std()

                learned_probabilities[feature] = {0:{'mean': class_0_mean, 'std': class_0_std},
                                                  1: {'mean': class_1_mean, 'std': class_1_std}}
                
        self.learned_params = learned_probabilities