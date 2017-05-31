#!/usr/bin/env python2.7
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
import argparse
import pdb
import csv
import imp
import math
import multiprocessing as mp
import types
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
ap.add_argument('--source-file', type=str, nargs='+', default=None,
                help="load dataframe(s) from source")
ap.add_argument('--out-file', type=str, default='./features.csv',
                help="where to output the resulting feature vector")
ap.add_argument('--buckets', type=int, default=0,
                help="bucket numeric values into n ordered categoricals")
ap.add_argument('--label', type=str, default='dropout',
                help="name of the label feature")
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


def estimate_median_private(arr, epsilon, low, high, target=0.5, splits=10,
                            buckets=2):
    """
    performs a private estimate of the median of an array via binary search
    todo: there's got to be research on this already
    """
    estimate = (high - low) / 2.
    step = estimate
    q = 1. / (1 + np.exp(epsilon))
    p = 1 - q
    left = arr[:]

    for i in range(splits):
        if len(left) == 0:
            break

        n = (len(left) + 1) / 2
        n = (len(arr) + splits) / splits

        part = left[:n]
        left = left[n:]
        step /= 2.

        above = 0
        for val in part:
            report_true = np.random.random() < p
            if val > estimate:
                if report_true:
                    above += 1
            else:
                if not report_true:
                    above += 1

        sample_frac = len([j for j in part if j > estimate]) / float(len(part))
        real_frac = len([j for j in arr if j > estimate]) / float(len(arr))
        est_frac = (above - q * len(part)) / (len(part) * (p - q))
        est_frac = min(max(est_frac, 0), 1)

        # calculate estimated margin of error
        # P[b' = 1] = P[b = 1] * p + P[b = 0] * q
        # stddev = sqrt(n * P[b'=1] * (1 - P[b'=1]))
        var_p = est_frac * p + (1 - est_frac) * q
        std = np.sqrt(len(part) * var_p * (1 - var_p)) / (len(part) * (p - q))

        print 'median %.3f: estimate = %3f +- %.3f, sample = %.3f, real = %.3f'%\
            (estimate, est_frac, std, sample_frac, real_frac)

        # if the estimated fraction of users who have
        if est_frac - std > target:
            estimate += step
        elif est_frac + std < target:
            estimate -= step
        else:
            continue

    return estimate


def bucket_data(df, label, buckets):
    # partition continuous and integer data into buckets
    for col in df.columns:
        do_buckets = col != label and (
            df[col].dtype == 'float64' or (
                df[col].dtype == 'int64' and len(set(df[col])) > buckets)
        )

        if not do_buckets:
            continue

        print 'bucketing column', repr(col)

        arr = np.nan_to_num(df[col].as_matrix())

        # this is here to mask out zeros, in case the majority of values are
        # zeros and it's impossible to do normal bucketing
        #mx = np.ma.masked_equal(arr, 0, copy=True)
        #bins = algos.quantile(arr[~mx.mask], np.linspace(0, 1, buckets+1))

        # then add back in a bucket specifically for zeros
        #bins = np.insert(bins, 0, 0)
        #bins[1] = bins[1] - bins[1] / 2

        median = estimate_median_private(arr, 2, min(arr), max(arr))
                                         #target=1. - float(sum(df[label])) / len(df[label]))

        epsilon = 1e-10
        bins = algos.quantile(arr, np.linspace(0, 1, buckets+1))
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + epsilon

        print bins[1], median
        bins[1] = median
        cuts = pd.tools.tile._bins_to_cuts(arr, bins, labels=range(buckets),
                                           include_lowest=True)

        df[col] = cuts
        continue

        #df[col] = pd.qcut(df[col], buckets, labels=range(buckets))

        # sample values until you get enough real ones. This doesn't work if
        # there are too many "NaN"s.
        #sample = df.sample(n=int(math.sqrt(len(df[col])) + 1))[col].copy()
        sample = df[col].sort_values(inplace=False, na_position='last')
        num_num = len(sample.dropna(inplace=False))
        n = float(num_num) / buckets

        # these are the percentiles of the numbers in the series - dictating
        # the boundaries of the buckets.

        # we only do this convoluted thing here to support the sampling step
        # above. Otherwise we would just sort everything and put elements
        # [n:i+n] into each bucket.
        bucket_list = [sample.iloc[int(i*n)] for i in range(1, buckets)]

        if False:    # simple method
            bucket_list = sorted(set(bucket_list))
            for i, row in df.iterrows():
                if np.isnan(row[col]):
                    val = 0
                else:
                    val = next((idx for idx, b in enumerate(bucket_list)
                                if b >= row[col]), len(bucket_list))
                df.set_value(i, col, val)

            print 'Bucket values for %s:' % col
            print '\tv <= %.3f' % bucket_list[0]
            for i in range(len(bucket_list) - 1):
                v = bucket_list[i]
                nv = bucket_list[i+1]
                print '\t%.3f < v <= %.3f' % (v, nv)
            print '\tv > %.3f' % bucket_list[-1]
            print

        else:       # more complicated method
            bucket_vals = {}
            exact_vals = {}

            idx = 0
            for v in sorted(list(set(bucket_list))):
                idx += 1
                bucket_vals[v] = idx
                if bucket_list.count(v) > 1:
                    idx += 1
                    exact_vals[v] = idx

            sorted_vals = sorted(bucket_vals.items())

            print 'Bucket values for %s:' % col
            print '\tv <= %.3f' % sorted_vals[0][0]
            for i in range(len(sorted_vals) - 1):
                v = sorted_vals[i][0]
                nv = sorted_vals[i+1][0]
                if v in exact_vals:
                    print '\tv == %.3f' % v

                if nv in exact_vals:
                    print '\t%.3f < v < %.3f' % (v, nv)
                else:
                    print '\t%.3f < v <= %.3f' % (v, nv)

            print '\tv > %.3f' % sorted_vals[-1][0]
            print

            num_buckets = len(bucket_vals) + len(exact_vals)
            for i, row in df.iterrows():
                if np.isnan(row[col]):
                    val = 0
                elif row[col] in exact_vals:
                    val = exact_vals[row[col]]
                else:
                    val = next((idx for b, idx in sorted_vals
                                if b >= row[col]), num_buckets)
                df.set_value(i, col, val)

        df[col] = df[col].astype(int)

    return df


def main():
    if args.source_file:
        dfs = []
        for path in args.source_file:
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs)
        label_name = args.label
    else:
        user_files = [f for f in listdir(args.data_dir) if
                      isfile(join(args.data_dir, f))]

        print "loading feature functions..."
        module = imp.load_source('feature_funcs', args.feature_funcs)
        global feature_funcs, label_func
        for func_name in dir(module):
            func = getattr(module, func_name)

            # only consider functions we find
            if type(func) == types.FunctionType:
                if func_name.startswith('ff_'):
                    feature_funcs.append(func)
                elif func_name.startswith('label_'):
                    label_func = func

        label_name = module.label_name
        print 'label:', label_name

        print "done."
        print "processing user files..."

        pool = mp.Pool(20)
        all_rows = pool.map(process_user_data, user_files)
        out_rows = filter(lambda r: r is not None, all_rows)
        df = pd.DataFrame(out_rows)

        print "done."
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
