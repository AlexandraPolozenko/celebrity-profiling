import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = stopwords.words('english')


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    
    text = " " + text + " "

    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(" ", text) # delete symbols which are in BAD_SYMBOLS_RE from text

    # delete stopwords from text
    for x in STOPWORDS:
        pattern = re.compile('\W' + x + '\W')
        text = pattern.sub(" ", text)

    return text

#
# def test_text_prepare():
#     examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
#                 "How to free c++ memory vector<int> * arr?"]
#     answers = ["sql server equivalent excels choose function",
#                "free c++ memory vectorint arr"]
#     for ex, ans in zip(examples, answers):
#         print(text_prepare(ex))
#         if text_prepare(ex) != ans:
#             return "Wrong answer for the case: '%s'" % ex
#     return 'Basic tests are passed.'
#
#
# print(test_text_prepare())
