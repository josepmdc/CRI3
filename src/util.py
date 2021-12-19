import pandas as pd

def load_dataset():
    dataset = pd.read_csv('data/FinalStemmedSentimentAnalysisDataset.csv', delimiter=';')
    dataset.drop(['tweetId', 'tweetDate'], axis=1, inplace=True)
    return dataset
 
def train_test_split(dataset, train_size=0.7, random_state=42):
    # We group by sentiment label and randomly select 'train_size' of each 
    # group, this way we have the same proportion of each class on each split
    train = dataset.groupby('sentimentLabel', group_keys=False).apply(
            lambda x: x.sample(frac=train_size, random_state=random_state))
    # For the test split we select all the elements we didn't select from the train split
    test = dataset.drop(train.index).to_numpy()
    train = train.to_numpy()
    return train[:,0], test[:,0], train[:,1].astype(int), test[:,1].astype(int)
