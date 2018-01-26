import numpy as np

import argparse
import sys
import arff

FLAGS = None;

# Helper function, turns the 'Y' / 'N' labels into numbers
def YN_to_num(x):
    if x == 'Y':
        return 1
    else:
        return 0

# Read the data
# TODO expand to read more data from more datasets
def read_data():
    return arff.load(open(FLAGS.data_dir + '/MW1.arff'))

# Preprocess the data
def preprocess_data(data):
    defect_col = len(data['attributes']) - 1

    features = np.array([x[0:defect_col] for x in data['data']], dtype='f')
    labels = np.reshape(np.array([YN_to_num(x[defect_col]) for x in data['data']]), (len(data['data']), 1))

    data_argmax = np.amax(np.array([x[0:defect_col] for x in data['data']]), axis=0)

    features = features / data_argmax

    return features, labels

def main():
    data = read_data()
    features, labels = preprocess_data(data)

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
