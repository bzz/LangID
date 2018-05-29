#!/usr/bin/env python
import os, sys
import codecs
import argparse

import pandas as pd
"""
Filter files from CSV for languages with small LoC

Input:  .csv \w columns: lang, abs_path, LoC
Output: .csv \w same columns
"""

DATASET_FILE = '../dataset-1/annotated_files_enry'
FREQ = 100

parser = argparse.ArgumentParser()
parser.add_argument("--files", type=str, default=DATASET_FILE + ".csv", help="a CSV file with a list of files")
parser.add_argument("--freq", type=int, default=FREQ, help="lowest bound of freq")
args = parser.parse_args()


def main():
    dst = DATASET_FILE + "_filtered.csv"
    print("Filtering files \w LoC < {} from {}", args.freq, args.files, dst)
    dataset = pd.read_csv(args.files, sep=';', names=['lang', 'file', 'LoC'])
    loc_freq = dataset.groupby(['lang'])['LoC'].sum().sort_values(ascending=False).reset_index(name="cnt")
    small_loc = loc_freq[loc_freq.cnt < args.freq]
    small_loc_files_num = dataset[dataset.lang.isin(small_loc['lang'].tolist())]['file'].nunique()
    print("Filtered {} of {} files".format(small_loc_files_num, dataset['file'].nunique()))
    print(small_loc)
    print("Saving results to {}".format(dst))
    new_dataset = dataset[~dataset.lang.isin(small_loc['lang'].tolist())]
    new_dataset.to_csv(path_or_buf=dst, header=False, index=False, sep=';', encoding="utf8")


if __name__ == '__main__':
    main()
