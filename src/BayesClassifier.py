import numpy as np
from collections import Counter

class BayesClassifier():
    def __init__(self, laplace=1, ignore_stopwords=False):
        self.positive_words = dict()
        self.negative_words = dict()
        self.laplace = laplace
        self.word_probabilities = dict()
        self.ignore_stopwords = ignore_stopwords

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
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        predictions = list()
        for text, sentiment in zip(X, y):
            positive = self.probability_positive
            negative = self.probability_negative
            for word in str(text).split():
                if word in self.word_probabilities:
                    negative *= self.word_probabilities[word][0]
                    positive *= self.word_probabilities[word][1]

            predicted_sentiment = int(positive > negative)

            # select the one with the highest prior in case of a tie
            if positive == negative:
                predicted_sentiment = int(self.probability_positive > self.probability_negative)

            predictions.append(predicted_sentiment)

            if predicted_sentiment == 1 and sentiment == 1:
                self.tp += 1
            elif predicted_sentiment == 1 and sentiment == 0:
                self.fp += 1
            elif predicted_sentiment == 0 and sentiment == 0:
                self.tn += 1
            elif predicted_sentiment == 0 and sentiment == 1:
                self.fn += 1

        return np.array(predictions)

    def classification_report(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        precision = self.tp / (self.tp + self.fp)
        negative_predictive_value = self.tn / (self.tn + self.fn)
        recall = self.tp / (self.tp + self.fn)
        specificity = self.tn / (self.tn + self.fp)
        f1_score = 2 / (1/recall + 1/precision)

        print(f'tp: {self.tp}, tn: {self.tn}, fp: {self.fp}, fn: {self.fn}')
        print(f'accuracy: {accuracy:.2f}')
        print(f'precision: {precision:.2f}')
        print(f'negative predictive value: {negative_predictive_value:.2f}')
        print(f'recall: {recall:.2f}')
        print(f'specificity: {specificity:.2f}')
        print(f'f1 score: {f1_score:.2f}')

    def __count_target_values(self):
        self.n_negatives = np.count_nonzero(self.y == 0)
        self.n_positives = np.count_nonzero(self.y == 1)

    def __compute_target_probabilities(self):
        total = self.n_positives + self.n_negatives
        self.probability_positive = self.n_positives / total
        self.probability_negative = self.n_negatives / total

    def __extract_words(self):
        """
        outcome:
            self.positive_words:  { "word" : occurances of word given sentiment is positive, ... }
            self.negative_words:  { "word" : occurances of word given sentiment is negative, ... }
        """
        neg = self.X[self.y == 0]
        pos = self.X[self.y == 1]
    
        counter = Counter()

        count_words = np.vectorize(
                lambda tweet: counter.update([
                    word.lower() for word in str(tweet).split()
                ]))

        count_words(pos)
        self.positive_words = dict(counter)
        
        counter = Counter() # reset the counter

        count_words(neg)
        self.negative_words = dict(counter)

    def __compute_probability_table(self):
        """
        outcome:
            self.word_probabilities: { 'word' : [P(word|negative), P(word|positive)], ... }
        """
        self.word_probabilities = dict()
        word_count = len(set().union(self.positive_words.keys(), self.negative_words.keys()))
        smoothing = word_count * self.laplace

        for word, freq in self.negative_words.items():
            self.word_probabilities.setdefault(word, [0, 0])[0] = \
                    (freq + self.laplace) / (self.n_negatives + smoothing)

        for word, freq in self.positive_words.items():
            self.word_probabilities.setdefault(word, [0, 0])[1] = \
                    (freq + self.laplace) / (self.n_positives + smoothing)

    def __load_stopwords(self):
        with open("data/stopwords") as file:
            self.stopwords = { word.strip(): True for word in file.readlines() }
