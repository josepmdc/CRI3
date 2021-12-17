import pandas as pd
from BayesClassifier import BayesClassifier

def load_dataset():
    return pd.read_csv('data/FinalStemmedSentimentAnalysisDataset.csv', delimiter=';')
 
def train_test_split(dataset, train_size=0.7, random_state=42):
    # We group by sentiment label and randomly select 'train_size' of each 
    # group, this way we have the same proportion of each class on each split
    train = dataset.groupby('sentimentLabel', group_keys=False).apply(
            lambda x: x.sample(frac=train_size, random_state=random_state))
    # For the test split we select all the elements we didn't select from the train split
    test = dataset.drop(train.index)
    return train, test


dataset = load_dataset()
train, test = train_test_split(dataset, train_size=0.7, random_state=42)

bayes = BayesClassifier(ignore_stopwords=True)
bayes.fit(train.drop('sentimentLabel', axis=1), train['sentimentLabel'])
bayes.predict(test.drop('sentimentLabel', axis=1), test['sentimentLabel'])
bayes.classification_report()
