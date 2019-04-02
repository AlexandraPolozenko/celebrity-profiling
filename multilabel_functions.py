
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import CategoricalEncoder
from sklearn.preprocessing import LabelBinarizer

import sklearn.metrics as m
import numpy as np


#tags_counts - dictionary tag - frequency
# mlb = MultiLabelBinarizer(classes=sorted(TAGS_COUNTS.keys()))
#y_train - training data tags
# TRAIN_TAGS = mlb.fit_transform(TRAIN_TAGS.values())
# print(TRAIN_TAGS)
#y_val - validation tags
#y_val = mlb.fit_transform(y_val)


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    return OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

#
# my_classifier = train_classifier(TRAIN_DATA, TRAIN_TAGS)
# tags_predicted = my_classifier.predict(TEST_DATA)


def work(X_train, y_train, tags_count, X_test, y_test, isBirthYear):
    # print(y_train)
    # print(y_test)
    mlb = MultiLabelBinarizer(classes=sorted(tags_count.keys()))
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.fit_transform(y_test)

    # y_test = list(y_test)
    # X_test_1 = list(X_test)
    # y_test = [y_test[0], y_test[2], y_test[3], y_test[4]]
    # X_test = [X_test_1[0], X_test_1[2], X_test_1[3], X_test_1[4]]
    # print(y_train)
    # print(y_test)
    # enc = LabelBinarizer
    # y_train = enc.fit_transform(np.asarray(list(y_train)))
    # y_test = enc.fit_transform(np.asarray(list(y_test)))
    # a = list(range(1940, 2012))
    # print(X_test)
    # le = LabelEncoder()
    # # le.fit(np.asarray(a))
    # y_train = le.fit_transform(np.asarray(list(y_train)))
    # y_test = le.fit_transform(np.asarray(list(y_test)))
    # print(le.get_params())
    #
    # print(y_train)
    # print(y_test)
    # enc = CategoricalEncoder(handle_unknown='ignore')
    # y_train = enc.fit_transform(np.asarray(list(y_train)))
    # y_test = enc.fit_transform(np.asarray(list(y_test)))

    print("binarizer done")

    classifier = train_classifier(X_train, y_train)

    print("classifier trained")

    y_val_predicted_labels = classifier.predict(X_test)
    y_val_predicted_scores = classifier.decision_function(X_test)

    # print(y_val_predicted_labels)

    print("tags predicted")

    y_val_pred_inversed = mlb.inverse_transform(
        y_val_predicted_labels)
    y_val_inversed = mlb.inverse_transform(y_test)

    for i in range(10):
        print('True labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            y_val_inversed[i],
            y_val_pred_inversed[i]
        ))

    if isBirthYear:
        i = 0
        yearToM = {}
        for i in range(1940, 1948):
            yearToM[i] = 9
        for i in range(1949, 1957):
            yearToM[i] = 8
        for i in range(1958, 1966):
            yearToM[i] = 7
        for i in range(1967, 1975):
            yearToM[i] = 6
        for i in range(1976, 1984):
            yearToM[i] = 5
        for i in range(1985, 1993):
            yearToM[i] = 4
        for i in range(1994, 2002):
            yearToM[i] = 3
        for i in range(2003, 2012):
            yearToM[i] = 2
        for i in range(10):
            pr = list(y_val_predicted_labels[i])[0]
            tr = list(y_val_inversed[i])[0]
            if 1940 <= pr <= 2012:
                M = yearToM[pr]
                if tr - M <= pr <= tr + M:
                    y_val_predicted_labels[i] = set(tr)

    f1_score = m.f1_score(y_test, y_val_predicted_labels, average='micro')
    return f1_score
