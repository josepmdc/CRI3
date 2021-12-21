import util
from BayesClassifier import BayesClassifier
import time

dataset = util.load_dataset()

print("==> 1. Testing Cross-Validation\n")
start = time.time()

bayes = BayesClassifier()
util.cross_validation_score(bayes, dataset)

end = time.time()
print(f"Runtime: {end - start:.2f} seconds\n")

print("==> 2.1 Testing different partition sizes\n")
for size in [0.60, 0.70, 0.80]:
    print(f"=== Train {size*100}%, Test size: {(100-size*100)}% ===\n")
    X_train, X_test, y_train, y_test = util.train_test_split(dataset, train_size=size, random_state=42)

    print(" == Without laplace smoothing ==")
    start = time.time()

    bayes = BayesClassifier(laplace=0)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds\n")

    print(" == With laplace smoothing ==")
    start = time.time()

    bayes = BayesClassifier()
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds\n")


X_train, X_test, y_train, y_test = util.train_test_split(dataset, train_size=0.7, random_state=42)

print("==> 2.2 Testing different dictionary sizes (Train: 70.0%, Test: 30.0%)\n")
for size in [100, 1000, 10000, None]: # if None size = all_words
    print(f"=== Dictinary Size {size} ===\n")

    print(" == Without laplace smoothing ==")
    start = time.time()

    bayes = BayesClassifier(laplace=0, dictionary_size=size)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()
    
    print(" == With laplace smoothing ==")
    start = time.time()

    bayes = BayesClassifier(dictionary_size=size)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds\n")


print("==> 2.3 Testing different partition sizes with fixed size dictionary (1000)\n")
for size in [0.60, 0.70, 0.80]:
    print(f"=== Train {size*100}%, Test size: {(100-size*100)}% ===\n")
    X_train, X_test, y_train, y_test = util.train_test_split(dataset, train_size=size, random_state=42)

    print(" == Without laplace smoothing ==")
    start = time.time()

    bayes = BayesClassifier(laplace=0, dictionary_size=1000)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()
    
    print(" == With laplace smoothing ==")
    start = time.time()
    
    bayes = BayesClassifier(dictionary_size=1000)
    bayes.fit(X_train, y_train)
    predictions = bayes.predict(X_test, y_test)
    bayes.classification_report()

    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds\n")
