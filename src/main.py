import pandas as pd
from collections import Counter

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
    returns:
        negative_words:  { "word" : freq_word given sentiment=negative, ... }
        positive_words:  { "word" : freq_word given sentiment=positive, ... }
    """
    neg, pos = [x for _, x in dataset.groupby('sentimentLabel')]

    positive_counts = Counter()
    pos['tweetText'].apply(lambda x: positive_counts.update([x for x in str(x).lower().split() if x[0] != '@']))
    # [x for x in str(x).lower().split() if x[0] != '@']
    negative_counts = Counter()
    neg['tweetText'].apply(lambda x: negative_counts.update([x for x in str(x).lower().split() if x[0] != '@']))

    return dict(positive_counts), dict(negative_counts)

def compute_probability_table(positive_words, negative_words, n_positives, n_negatives):
    """
    returns a dictionary like this:
    { 'word' : [P(word|negative), P(word|positive)], ... }
    """
    probabilities = dict()
    for word, freq in negative_words.items():
        probabilities.setdefault(word, [0, 0])[0] = (freq + 1) / (n_negatives + 2)
    for word, freq in positive_words.items():
        probabilities.setdefault(word, [0, 0])[1] = (freq + 1) / (n_positives + 2)
    return probabilities

def predict():
    pass

def main():
    dataset = load_dataset()

    train, test = train_test_split(dataset, train_size=0.7, random_state=42)
    
    positive_words, negative_words = extract_words(train)

    counts = dataset['sentimentLabel'].value_counts()
    n_negatives = counts[0]
    n_positives = counts[1]

    word_probabilities = compute_probability_table(positive_words, negative_words, n_positives, n_negatives)

    total = n_positives + n_negatives
    probability_positive = n_positives / total
    probability_negative = n_negatives / total

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for tweet_text, tweet_sentiment in zip(test['tweetText'], test['sentimentLabel']):
        positive = probability_positive
        negative = probability_negative
        for word in str(tweet_text).split():
            if word in word_probabilities:
                negative *= word_probabilities[word][0]
                positive *= word_probabilities[word][1]

        sentiment = int(positive > negative)
        
        # select the one with the highest prior in case of a tie
        if positive == negative:
            sentiment = int(probability_positive > probability_negative)

        if sentiment == 1 and tweet_sentiment == 1:
            tp += 1
        elif sentiment == 1 and tweet_sentiment == 0:
            fp += 1
        elif sentiment == 0 and tweet_sentiment == 0:
            tn += 1
        elif sentiment == 0 and tweet_sentiment == 1:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    negative_predictive_value = tn / (tn + fn)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = 2 / (1/recall + 1/precision)

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Negative predictive value: {negative_predictive_value:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print(f'F1 score: {f1_score:.2f}')

main()
