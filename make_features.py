import pandas as pd
import numpy as np
import argparse
import pdb
import csv
import imp
from collections import defaultdict
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
ap.add_argument('--out-file', type=str, default='./features.csv',
                help="where to output the resulting feature vector")
args = ap.parse_args()

num_yes = 0
num_no = 0
num_null = 0

# accepts a dataframe of user data, sorted by time, and calls feature functions
# on the data to generate a dict of features
def process_user_data(user, df, label_func, feature_funcs):
    global num_yes, num_no, num_null

    # compute the label
    label_name, label_val = label_func(df, pred_time=args.pred_time,
                                       lead_time=args.lead_time)
    if label_val is None:
        num_null += 1
        return None

    # counters
    if label_val:
        num_yes += 1
    else:
        num_no += 1

    # add the label as the first feature
    feature_names = [label_name]
    feature_vals = {label_name: label_val}

    # filter out data we can't know about yet
    df = df[df[args.time_index] <= args.pred_time - args.lead_time]

    # not sure what negative week values mean. before start of class?
    df = df[df[args.time_index] >= 0]

    # compute each feature function
    for func in feature_funcs:
        features = func(df)
        for name, val in features:
            feature_names.append(name)
            feature_vals[name] = val

    return feature_names, feature_vals


# count the frequency of negative week values in the dataset
# this is a one-off function, not really useful for anything
def count_neg_weeks(user_files):
    neg_wks = defaultdict(int)
    for uf in user_files:
        user = uf.replace('.csv', '')
        with open(join(args.data_dir, uf)) as f:
            data = pd.read_csv(f)

        df = data[data.week < 0]
        if len(df.week):
            for i in df.week:
                neg_wks[i] += 1

    for item in sorted(neg_wks.items(), key=lambda w: w[1]):
        print item


def main():
    user_files = [f for f in listdir(args.data_dir) if
                  isfile(join(args.data_dir, f))]

    print "loading feature functions..."
    feature_funcs = []
    module = imp.load_source('feature_funcs', args.feature_funcs)
    for func_name in dir(module):
        if func_name.startswith('ff_'):
            feature_funcs.append(getattr(module, func_name))
        elif func_name.startswith('label_'):
            label_func = getattr(module, func_name)

    print "done."
    print "processing user files..."
    out_rows = []
    pct = 5.0
    for i, uf in enumerate(user_files):
        complete = (float(i) / len(user_files)) * 100.0
        if complete > pct:
            print '%.0f%% complete' % complete
            pct += 5.0

        user = uf.replace('.csv', '')
        with open(join(args.data_dir, uf)) as f:
            df = pd.read_csv(f)

        processed = process_user_data(user, df, label_func, feature_funcs)
        if processed is not None:
            features, row = processed
            out_rows.append(row)

    print "done."
    print "writing to csv..."

    with open(args.out_file, 'w') as outfile:
        writer = csv.DictWriter(outfile, features)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print "done."
    print "num_yes =", num_yes
    print "num_no =", num_no
    print "num_null =", num_null

if __name__ == '__main__':
   main()
