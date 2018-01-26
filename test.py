import numpy as np

import argparse
import sys
import arff

FLAGS = None;

def read_data():
    return arff.load(open(FLAGS.data_dir + '/MW1.arff'))

def main():
    print("hello\n");
    data = read_data()


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
