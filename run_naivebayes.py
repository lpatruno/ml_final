"""
Driver to run the Naive Bayes model analysis.

author: Luigi Patruno
"""
import numpy as np
from get_data import data
from sklearn.cross_validation import train_test_split

from naivebayes import NaiveBayes

def main():
    
    df = data()
    
    for train_size in np.linspace(.5, .9, 5):
    
        train, test = train_test_split(df, train_size=train_size, random_state=42)

        # Since there is only 1 sample with native-country == Holand-Netherlands,
        # ensure that this sample is in the training set
        if 'Holand-Netherlands' in test['native-country'].unique():
            train = train.append( test[test['native-country'] == 'Holand-Netherlands'] )
            test = test[ test['native-country'] != 'Holand-Netherlands']

        for ignore_missing in [True, False]:
            nb = NaiveBayes(ignore_missing=ignore_missing)
            nb.learn_parameters(train)
            acc = nb.score(test[ test['native-country'] != 'Holand-Netherlands'])

            print('\nTrain size: {} Test error: {} Ignore features with missing values: {}'.format(train_size, (1-acc), ignore_missing))
            
if __name__ == '__main__':
    main()