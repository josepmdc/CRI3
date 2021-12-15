import pandas as pd

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

def extract_words(dataset):
    """
    returns a dictionary like this:
    { 
        'word' : {
            'positive' : count_of_positive,
            'negative' : count_of_negative
        }
    }
    """
    words = dict()
    n_positives = 0
    n_negatives = 0
    for _, tweet in dataset.iterrows():
        for word in str(tweet['tweetText']).split():
            words[word] = words.setdefault(word, { 'positive' : 0, 'negative' : 0 })
            if tweet['sentimentLabel'] == 0:
                words[word]['negative'] += 1
                n_negatives += 1
            else:
                words[word]["positive"] += 1
                n_positives += 1
    return words, n_positives, n_negatives

def compute_probability_table(words, n_positives, n_negatives):
    """
    returns a dictionary like this:
    { 
        'word' : {
            'positive' : P(word|positive),
            'negative' : P(word|negative)
        }
    }
    """
    return dict(map(lambda word: (word[0], {
        "positive": word[1]["positive"] / n_positives,
        "negative": word[1]["negative"] / n_negatives 
        }), words.items()))

def main():
    dataset = load_dataset()
    train, _ = train_test_split(dataset, train_size=0.7, random_state=42)
    words, n_positives, n_negatives = extract_words(train)
    print(compute_probability_table(words, n_positives, n_negatives))

main()
