import numpy as np

import argparse
import sys
import arff
import pprint

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

FLAGS = None;
pp = pprint.PrettyPrinter(indent=2)

# Helper function, turns the 'Y' / 'N' labels into numbers
def YN_to_num(x):
    if x == 'Y':
        return 1
    else:
        return 0

# Read the data
# TODO expand to read more data from more datasets
def read_data():
    PC1 = arff.load(open(FLAGS.data_dir + '/PC1.arff'))
    PC2 = arff.load(open(FLAGS.data_dir + '/PC2.arff'))
    PC3 = arff.load(open(FLAGS.data_dir + '/PC3.arff'))
    PC4 = arff.load(open(FLAGS.data_dir + '/PC4.arff'))

    # Need to remove LOC_BLANK attrib. from PC{1, 3, 4} since PC2 lacks that one
    PC2['data'] += [x[1:] for x in PC1['data']]
    PC2['data'] += [x[1:] for x in PC3['data']]
    PC2['data'] += [x[1:] for x in PC4['data']]

    return PC2

# Preprocess the data
# Split into features and labels, normalize features to [0, 1]
def preprocess_data(data):
    label_col = len(data['attributes']) - 1

    features = np.array([x[0:label_col] for x in data['data']], dtype='f')
    labels = np.array([YN_to_num(x[label_col]) for x in data['data']])

    data_argmax = np.amax(np.array([x[0:label_col] for x in data['data']]), axis=0)

    #features = features / data_argmax

    return features, labels

def test_helper(fun, test_name, labels, pred):
    print(test_name, fun(labels, pred))


# Trains and tests the speciifed classifier on the dataset
def train_and_test(clf, dataset, clf_name=None):
    clf.fit(dataset['X_train'], dataset['Y_train'])
    pred = clf.predict(dataset['X_test'])

    if clf_name:
        print("Classifier:\t{}".format(clf_name))
        print("---")

    print("Accuracy:\t{}".format(accuracy_score(dataset['Y_test'], pred)))
    print("Precision:\t{}".format(precision_score(dataset['Y_test'], pred)))
    print("Recall:\t\t{}".format(recall_score(dataset['Y_test'], pred)))
    print()

def main():
    dataset = dict.fromkeys(['X_train', 'X_test', 'Y_train', 'Y_test'])
    train = {}
    test = {}

    data = read_data()
    features, labels = preprocess_data(data)
    total_defects = sum(labels)

    dataset['X_train'], dataset['X_test'], dataset['Y_train'], dataset['Y_test'] = train_test_split(features, labels, test_size = 0.33)

    train_defects = sum(dataset['Y_train'])
    test_defects = sum(dataset['Y_test'])

    print("Defects in dataset:\t\t{}".format(total_defects))
    print("Defects in training set:\t{} ({}%)".format(train_defects, train_defects / total_defects * 100))
    print("Defects in test set:\t\t{} ({}%)".format(test_defects, test_defects / total_defects * 100))
    print()
    train_and_test(svm.SVC(), dataset, clf_name = "Support Vector Machine")
    train_and_test(GaussianNB(), dataset, clf_name = "Naive Bayes")

    # Set up run arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data-dir',
            type=str,
            default='',
            help="Directory of the data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
