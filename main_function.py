from json_preprocessing import read_training_tags, read_training_tweets
from tf_idf_functions import tfidf_features
from multilabel_functions import work


def main():
    X_train, X_test = read_training_tweets()
    print("X sets done\n")
    print("X_test\n")
    print(X_test)
    print("\n")

    y_train_occupation, y_train_gender, y_train_fame, y_train_birthyear, \
    y_test_occupation, y_test_gender, y_test_fame, y_test_birthyear, \
    tags_occupation, tags_gender, tags_fame, tags_birthyear = read_training_tags(X_train, X_test)

    X_train = X_train.values()
    X_test = X_test.values()

    print("y and tags done\n")

    X_train, X_test, vocab = tfidf_features(X_train, X_test)
    print("tfidf build")

    f1_occupation = work(X_train, y_train_occupation, tags_occupation, X_test, y_test_occupation, False)
    print("occupation done")
    print(f1_occupation)
    f1_gender = work(X_train, y_train_gender, tags_gender, X_test, y_test_gender, False)
    print("gender done")
    print(f1_gender)
    f1_fame = work(X_train, y_train_fame, tags_fame, X_test, y_test_fame, False)
    print("fame done")
    print(f1_fame)
    f1_birthyear = work(X_train, y_train_birthyear, tags_birthyear, X_test, y_test_birthyear, True)
    print("birthyear done")
    print(f1_birthyear)

    if f1_occupation == 0:
        f1_occupation = 0.00001
    if f1_gender == 0:
        f1_gender = 0.00001
    if f1_fame == 0:
        f1_fame = 0.00001
    if f1_birthyear == 0:
        f1_birthyear = 0.00001

    cRank = 4 / ((1 / f1_occupation) + (1 / f1_gender) + (1 / f1_fame) + (1 / f1_birthyear))
    print(cRank)


print("started")
main()

