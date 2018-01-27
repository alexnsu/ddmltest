import numpy as np

import argparse
import sys
import arff
import pprint

from sklearn import svm

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
def preprocess_data(data):
    label_col = len(data['attributes']) - 1

    features = np.array([x[0:label_col] for x in data['data']], dtype='f')
    labels = np.array([YN_to_num(x[label_col]) for x in data['data']])

    data_argmax = np.amax(np.array([x[0:label_col] for x in data['data']]), axis=0)

    features = features / data_argmax

    return features, labels

def main():
    data = read_data()
    features, labels = preprocess_data(data)
    clf = svm.SVC()
    clf.fit(features, labels)

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
