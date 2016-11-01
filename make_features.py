import pandas as pd
import numpy as np
import argparse
import pdb
import csv
from collections import defaultdict
from os import listdir
from os.path import isfile, join

ap = argparse.ArgumentParser()
ap.add_argument('data_dir', type=str, help='path to the CSV user data')
ap.add_argument('--label-feat', type=str, default='dropout_1',
                help='feature for the label we are trying to predict')
ap.add_argument('--label-val', type=str, default=0,
                help='label value we are looking for')
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

# `user` is a string identifier, `df` is a DataFrame
def process_user_data(user, df):
    global num_yes, num_no, num_null
    df = df[df.week >= 0]

    # find the first week after our time cutoff, and the last week before it.
    next_week = df[df.week >= args.pred_time].iloc[0]
    last_week = df[df.week <= args.pred_time - args.lead_time].iloc[-1]

    # if someone has already dropped out before our cutoff, ignore their data.
    if last_week[args.label_feat] == args.label_val:
        num_null += 1
        return None

    #last_week = df[df[args.label_feat] == args.label_val].iloc[0]['week']

    # filter out data we can't know yet
    df = df[df.week <= args.pred_time - args.lead_time]

    features = list(df.columns)
    features.remove('week')
    fdict = {}

    for f in list(features):
        new_feat = f + '_average'
        features.append(new_feat)
        last_week_val = df.iloc[-1][f]

        # take the label value from the first week after our cutoff
        if f == args.label_feat:
            label = next_week[args.label_feat]
            if label == args.label_val:
                num_yes += 1
            else:
                num_no += 1
            fdict[f] = label
        else:
            fdict[f] = last_week_val
            fdict[new_feat] = np.mean(df[f])

    return features, fdict


# count the frequency of negative week values in the dataset
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
            data = pd.read_csv(f)

        processed = process_user_data(user, data)
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
