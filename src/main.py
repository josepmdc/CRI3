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
    { 'word' : [count_of_positive, count_of_negative], ... }
    """
    pos, neg = [x for _, x in df.groupby(df['sentimentLabel'] == 0)]

    positive_counts = Counter()
    pos['tweetText'].apply(lambda x: positive_counts.update(x.lower().split()))

    negative_counts = Counter()
    neg['tweetText'].apply(lambda x: negative_counts.update(x.lower().split()))

    return dict(positive_counts), dict(negative_counts)

    # words = dict()
    # n_positives = 0
    # n_negatives = 0
    # for _, tweet in dataset.iterrows():
    #     for word in str(tweet['tweetText']).split():
    #         sentiment = tweet['sentimentLabel']
    #         words[word] = words.setdefault(word, [0, 0])
    #         words[word][sentiment] += 1
    # return words

def compute_probability_table(words, n_positives, n_negatives):
    """
    returns a dictionary like this:
    { 'word' : [P(word|negative), P(word|positive)], ... }
    """
    return dict(map(lambda word: 
            (word[0], [word[1][0] / n_negatives, word[1][1] / n_positives]), 
        words.items()))

def main():
    dataset = load_dataset()

    counts = dataset['sentimentLabel'].value_counts()
    n_negatives = counts[0]
    n_positives = counts[1]

    train, test = train_test_split(dataset, train_size=0.7, random_state=42)
    
    positive_words, negative_words = extract_words(train)
    word_probabilities = compute_probability_table(positive_words, negative_words, n_positives, n_negatives)
    
    total = n_positives + n_negatives
    probability_positive = n_positives / total
    probability_negative = n_negatives / total

    tp = 0
    tn = 0

    for _, tweet in test.iterrows():
        positive = probability_positive
        negative = probability_negative
        for word in str(tweet['tweetText']).split():
            negative *= word_probabilities.setdefault(word, [0, 0])[0]
            positive *= word_probabilities.setdefault(word, [0, 0])[1]

        sentiment = int(positive > negative)
        if sentiment == tweet['sentimentLabel']:
            tp += 1
        else:
            tn += 1

    print(f'TP: {tp}, TN: {tn}')

main()
