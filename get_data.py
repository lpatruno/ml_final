"""
Function to load the adult data set data and create the 0,1 class labels
"""
import pandas as pd

def data():
    """
    Returns the adult data set with 0,1 class labels ina pandas DataFrame
    """
    
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
          'hours-per-week', 'native-country', 'label']
    
    df = pd.read_csv('adult.data', names=columns, skipinitialspace=True)
    
    # Use 0 and 1 as class labels
    df.loc[ df['label'] == '<=50K', 'class'] = 0
    df.loc[ df['label'] == '>50K', 'class'] = 1
    
    return df

if __name__ == '__main__':
    
    df = data()
    
    print 'Data loaded with columns \n{}'.format(df.columns)
    
    # Print missing data statistics
    for col in df:
        if '?' in df[col].unique():
            num_missing = df[df[col] == '?'].shape[0]
            pct_missing = num_missing / float(df.shape[0])
            print '{} values missing ({} %) in column {}'.format(num_missing, 100*pct_missing, col)