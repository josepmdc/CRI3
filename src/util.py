import pandas as pd
import numpy as np

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

# Adaptat de la pràctica 2
def cross_validation_score(model, dataset, n_folds=5):
    folds = np.array_split(dataset.to_numpy(), n_folds)
    for k in range(n_folds):
        train = folds.copy()
        test = train[k]
        train = np.delete(np.array(train, dtype=object), k, axis=0)
        train = np.concatenate(train, axis=0)
        
        X_train, X_test, y_train, y_test = train[:,0], test[:,0], train[:,1], test[:,1]

        model.fit(X_train, y_train)
        model.predict(X_test, y_test)
        score = model.score()
        print(f"Score for fold {k + 1}: {score}")
    print()
