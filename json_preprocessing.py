import json
from text_formating_functions import text_prepare


def read_training_tweets():
    train_data = {}
    test_data = {}
    file = open("data/feeds.ndjson", "r")
    flag = 0

    while True:
        x = file.readline()

        # if flag == 5000:
        #     break

        if (flag % 1000) == 0:
            print("1000 id's read")

        if x == '\n':
            continue

        if x == '':
            break

        js = json.loads(x)

        tx = ''
        for y in js['text']:
            a = text_prepare(y)
            tx += a

        if flag < 10:
            test_data[js['id']] = tx
        else:
            train_data[js['id']] = tx

        flag += 1

    return train_data, test_data


def read_training_tags(X_train, X_test):
    train_occupation = {}
    train_gender = {}
    train_fame = {}
    train_birthyear = {}
    test_occupation = {}
    test_gender = {}
    test_fame = {}
    test_birthyear = {}

    tags_occupation = {'sports': 0, 'performer': 0, 'creator': 0,
                   'politics': 0, 'manager': 0, 'science': 0, 'professional': 0, 'religious': 0}
    tags_gender = {'male': 0, 'female': 0, 'nonbinary': 0}
    tags_fame = {'rising': 0, 'star': 0, 'superstar': 0}
    tags_birthyear = {}

    for a in range(1940, 2012):
        tags_birthyear[a] = 0

    file = open("data/labels.ndjson", "r")

    while True:
        x = file.readline()

        if x == '\n':
            continue

        if x == '':
            break

        oc = set()
        gen = set()
        fam = set()
        by = set()

        js = json.loads(x)
        ids = js['id']
        oc.add(js['occupation'])
        gen.add(js['gender'])
        fam.add(js['fame'])
        by.add(js['birthyear'])
        tags_occupation[js['occupation']] += 1
        tags_gender[js['gender']] += 1
        tags_fame[js['fame']] += 1
        tags_birthyear[js['birthyear']] += 1

        if ids in X_train:
            train_occupation[ids] = oc
            train_gender[ids] = gen
            train_fame[ids] = fam
            train_birthyear[ids] = by
        if ids in X_test:
            test_occupation[ids] = oc
            test_gender[ids] = gen
            test_fame[ids] = fam
            test_birthyear[ids] = by

    return train_occupation.values(), train_gender.values(), train_fame.values(), train_birthyear.values(), \
           test_occupation.values(), test_gender.values(), test_fame.values(), test_birthyear.values(), \
           tags_occupation, tags_gender, tags_fame, tags_birthyear


# read_training_tweets()
# read_training_tags()
# print(TRAIN_DATA)
# print(TRAIN_TAGS)