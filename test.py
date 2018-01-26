import numpy as np

import argparse
import sys
import arff

FLAGS = None;

# Read the data
# TODO expand to read more data from more datasets
def read_data():
    return arff.load(open(FLAGS.data_dir + '/MW1.arff'))

# Splits the data and normalizes it
def split_data(data):
    defect_col = len(data['attributes']) - 1

    correct = np.array([x[0:defect_col] for x in data['data'] if x[defect_col] == 'N'], dtype='f')
    defects = np.array([x[0:defect_col] for x in data['data'] if x[defect_col] == 'Y'], dtype='f')

    if not len(defects) + len(correct) == len(data['data']):
        raise AssertionError('Data did not split properely!')

    data_argmax = np.amax(np.array([x[0:defect_col] for x in data['data']]), axis=0)

    correct = correct / data_argmax
    defects = defects / data_argmax

    return correct, defects

# Run the stuff
def main():
    data = read_data()
    correct, defects = split_data(data)

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
