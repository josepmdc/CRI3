import util
from BayesClassifier import BayesClassifier

dataset = util.load_dataset()

print("==> Testing different partition sizes\n")
for size in [0.60, 0.70, 0.80]:
    print(f"=== Train {size*100}%, Test size: {(100-size*100)}% ===\n")
    X_train, X_test, y_train, y_test = util.train_test_split(dataset, train_size=size, random_state=42)

    bayes = BayesClassifier()
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()

    print("Without laplace smoothing")
    bayes = BayesClassifier(laplace=0)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()


X_train, X_test, y_train, y_test = util.train_test_split(dataset, train_size=0.7, random_state=42)

print("==> Testing different dictionary sizes\n")
for size in [100, 1000, 10000, None]: # if None size = all_words
    print(f"=== Dictinary Size {size} ===\n")
    bayes = BayesClassifier(dictionary_size=size)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()
