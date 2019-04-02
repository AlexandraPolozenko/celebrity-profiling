from sklearn.feature_extraction.text import TfidfVectorizer

from text_formating_functions import text_prepare


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


def tfidf_features(X_train, X_test):
    """
        X_train, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train and test sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')  # '(\S+)'  means any no white space
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_test, tfidf_vectorizer.vocabulary_


# tfidf_vectorizer.vocabulary_ returns just index of feature
#
# i = 0
# for x in corpus:
#     a = text_prepare(x)
#     print(a)
#     corpus[i] = a
#     i += 1
#
# vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern='(\S+)')
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(X.shape)
# print(vectorizer.get_stop_words())
#
# print(X)