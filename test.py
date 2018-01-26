import numpy as np

import argparse
import sys
import arff

FLAGS = None;

def read_data():
    return arff.load(open(FLAGS.data_dir + '/MW1.arff'))

def split_data(data):
    defect_col = len(data['attributes']) - 1

    defects = [x for x in data['data'] if x[defect_col] == 'Y']
    correct = [x for x in data['data'] if x[defect_col] == 'N']

    if not len(defects) + len(correct) == len(data['data']):
        raise AssertionError('Data did not split properely!')

    return correct, defects

def main():
    data = read_data()

    correct, defects = split_data(data)

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
