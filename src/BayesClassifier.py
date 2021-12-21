import numpy as np
from collections import Counter

class BayesClassifier():
    def __init__(self, laplace=1, ignore_stopwords=True, dictionary_size=None):
        self.positive_words = dict()
        self.negative_words = dict()
        self.laplace = laplace
        self.word_probabilities = dict()
        self.ignore_stopwords = ignore_stopwords
        self.dictionary_size = dictionary_size
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.__load_stopwords()
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_features = len(X)

        self.__count_target_values()
        self.__compute_target_probabilities()
        self.__extract_words()
        self.__compute_probability_table()
     
    def predict(self, X, y):
        predictions = list()

        for text, actual_sentiment in zip(X, y):
            positive = self.probability_positive
            negative = self.probability_negative
            for word in str(text).split():
                if word in self.word_probabilities:
                    negative *= self.word_probabilities[word][0]
                    positive *= self.word_probabilities[word][1]

            predicted_sentiment = int(positive >= negative)
            predictions.append(predicted_sentiment)
            self.__update_score(predicted_sentiment, actual_sentiment)

        return np.array(predictions)

    def classification_report(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        precision = self.tp / (self.tp + self.fp)
        negative_predictive_value = self.tn / (self.tn + self.fn)
        recall = self.tp / (self.tp + self.fn)
        specificity = self.tn / (self.tn + self.fp)

        print(f'tp: {self.tp},    fp: {self.fp}') 
        print(f'fn: {self.fn},   tn: {self.tn}')
        print('--------------------------------------------')
        print(f'accuracy:                  {accuracy}')
        print(f'precision:                 {precision}')
        print(f'negative predictive value: {negative_predictive_value}')
        print(f'recall:                    {recall}')
        print(f'specificity:               {specificity}')
        print()

    def __count_target_values(self):
        self.n_negatives = np.count_nonzero(self.y == 0)
        self.n_positives = np.count_nonzero(self.y == 1)

    def __compute_target_probabilities(self):
        total = self.n_positives + self.n_negatives
        self.probability_positive = (self.n_positives + self.laplace) / (total + self.laplace * 2)
        self.probability_negative = (self.n_negatives + self.laplace) / (total + self.laplace * 2)

    def __extract_words(self):
        """
        outcome:
            self.positive_words:  { "word" : occurances of word given sentiment is positive, ... }
            self.negative_words:  { "word" : occurances of word given sentiment is negative, ... }
        """
        neg = self.X[self.y == 0]
        pos = self.X[self.y == 1]
    
        count_words = np.vectorize(
                lambda tweet, counter: counter.update([
                    word.lower() for word in str(tweet).split()
                    if word not in self.stopwords
                ]))

        counter = Counter()
        count_words(pos, counter)
        self.positive_words = dict(counter.most_common(self.dictionary_size))
        
        counter = Counter()
        count_words(neg, counter)
        self.negative_words = dict(counter.most_common(self.dictionary_size))

    def __compute_probability_table(self):
        """
        outcome:
            self.word_probabilities: { 'word' : [P(word|negative), P(word|positive)], ... }
        """
        self.word_probabilities = dict()
        word_count = len(set().union(self.positive_words.keys(), self.negative_words.keys()))

        zero_probability = 0 if self.laplace == 0 else self.laplace / (word_count * self.laplace)
        default_probability = [zero_probability, zero_probability]

        for word, freq in self.negative_words.items():
            self.word_probabilities.setdefault(word, default_probability)[0] = \
                    (freq + self.laplace) / (self.n_negatives + word_count * self.laplace)

        for word, freq in self.positive_words.items():
            self.word_probabilities.setdefault(word, default_probability)[1] = \
                    (freq + self.laplace) / (self.n_positives + word_count * self.laplace)

    def __load_stopwords(self):
        with open("data/stopwords") as file:
            self.stopwords = { word.strip(): True for word in file.readlines() }

    def __update_score(self, predicted_sentiment, actual_sentiment):
        if predicted_sentiment == 1 and actual_sentiment == 1:
            self.tp += 1
        elif predicted_sentiment == 1 and actual_sentiment == 0:
            self.fp += 1
        elif predicted_sentiment == 0 and actual_sentiment == 0:
            self.tn += 1
        elif predicted_sentiment == 0 and actual_sentiment == 1:
            self.fn += 1
