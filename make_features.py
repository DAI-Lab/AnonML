import pandas as pd
import numpy as np
import argparse
import pdb
import csv
import imp
import math
import multiprocessing as mp
from collections import defaultdict, OrderedDict
from os import listdir
from os.path import isfile, join


ap = argparse.ArgumentParser()
ap.add_argument('data_dir', type=str, help='path to the CSV user data')
ap.add_argument('--feature-funcs', type=str, default='./edx_feature_funcs.py',
                help="path to python file defining feature functions")
ap.add_argument('--time-index', type=str, default='week',
                help="name of the time index column in the dataframe")
ap.add_argument('--pred-time', type=int, default=5,
                help="week in which we make a prediction")
ap.add_argument('--lead-time', type=int, default=1,
                help="number of weeks ahead we're trying to predict a label")
ap.add_argument('--source-file', type=str, default=None,
                help="load dataframe from source")
ap.add_argument('--out-file', type=str, default='./features.csv',
                help="where to output the resulting feature vector")
ap.add_argument('--buckets', type=int, default=0,
                help="bucket numeric values into n ordered categoricals")
args = ap.parse_args()

label_func = None
feature_funcs = []

# accepts a dataframe of user data, sorted by time, and calls feature functions
# on the data to generate a dict of features
def process_user_data(user_file):
    # load user data as a Pandas DataFrame
    with open(join(args.data_dir, user_file)) as f:
        df = pd.read_csv(f)

    # compute the label
    label_name, label_val = label_func(df, pred_time=args.pred_time,
                                            lead_time=args.lead_time)
    if label_val is None:
        return None

    # add the label as the first feature
    features = OrderedDict()
    features[label_name] = label_val

    # filter out data we can't know about yet
    df = df[df[args.time_index] <= args.pred_time - args.lead_time]

    # not sure what negative week values mean. before start of class?
    df = df[df[args.time_index] >= 0]

    # compute each feature function
    for func in feature_funcs:
        features.update(func(df))

    return features

def bucket_data(df, label, buckets):
    # partition continuous and integer data into buckets
    for col in df.columns:
        if col == label or df[col].dtype != 'float64':
            continue
        #df[col] = pd.qcut(df[col], buckets, labels=range(buckets))

        # sample values until you get enough real ones. This doesn't work if
        # there are too many "NaN"s.
        #sample = df.sample(n=int(math.sqrt(len(df[col])) + 1))[col].copy()
        sample = df[col].sort_values(inplace=False, na_position='last')
        num_not_na = len(sample.dropna(inplace=False))
        n = num_not_na / buckets

        # these are the percentiles of the numbers in the series - dictating
        # the boundaries of the buckets.
        bucket_vals = [sample.iloc[i] for i in range(n-1, num_not_na, n)]

        # we only do this convoluted thing here to support the sampling step
        # above. Otherwise we would just sort everything and put elements
        # [n:i+n] into each bucket.
        for i, row in df.iterrows():
            if np.isnan(row[col]):
                val = 0
            else:
                val = next((j+1 for j, b in enumerate(bucket_vals)
                            if b >= row[col]), buckets)
            df.set_value(i, col, val)
        df[col] = df[col].astype(int)

        # print out the values of the bucket delimiters
        print col, ', '.join('%.2f' % b for b in bucket_vals)

    return df


def main():
    if args.source_file:
        df = pd.read_csv(args.source_file)
        label_name = df.columns[0]
    else:
        user_files = [f for f in listdir(args.data_dir) if
                      isfile(join(args.data_dir, f))]

        print "loading feature functions..."
        module = imp.load_source('feature_funcs', args.feature_funcs)
        global feature_funcs, label_func
        for func_name in dir(module):
            if func_name.startswith('ff_'):
                feature_funcs.append(getattr(module, func_name))
            elif func_name.startswith('label_'):
                label_func = getattr(module, func_name)

        print "done."
        print "processing user files..."

        pool = mp.Pool(8)
        all_rows = pool.map(process_user_data, user_files)
        out_rows = filter(lambda r: r is not None, all_rows)
        df = pd.DataFrame(out_rows)

        print "done."

        label_name = df.columns[0]
        num_yes = np.sum(df[label_name])
        num_no = len(df[label_name]) - num_yes
        num_null = len(all_rows) - len(out_rows)
        print "num_yes =", num_yes
        print "num_no =", num_no
        print "num_null =", num_null

    if args.buckets > 0:
        print "bucketing data..."
        df = bucket_data(df, label_name, args.buckets)
        print "done."

    print "writing to csv..."
    with open(args.out_file, 'w') as outfile:
        df.to_csv(outfile, index=False)
    print "done."


if __name__ == '__main__':
   main()
