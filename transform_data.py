from collections import defaultdict
import csv
import json
import argparse
import pandas as pd
import numpy as np
import os

ap = argparse.ArgumentParser()
ap.add_argument('data_file', type=str, help='path to the raw data file')
ap.add_argument('--output-path', type=str, default='./',
                help='where to save output data')
args = ap.parse_args()

def main():
    jsf = open('feature_map.json')
    fmap = json.load(jsf)
    jsf.close()

    print 'loading in data...'
    f = open(args.data_file)
    reader = csv.DictReader(f)
    rows = [r for r in reader]
    f.close()
    print 'done.'

    print 'gathering all users and features...'
    # collect all rows and all columns for each user
    users = {}
    features = set()
    for row in rows:
        user = row['user_id']
        feature_id = row['longitudinal_feature_id']
        week = int(row['longitudinal_feature_week'])

        features.add(feature_id)

        if user not in users:
            users[user] = set()
        users[user].add(week)

    print 'done.'
    print 'generating user dataframes...'
    data = {}
    max_weeks = 0
    for user, weeks in users.iteritems():
        cols = {('%s_%s' % (fmap[fid], fid) if fid in fmap else fid):
                    [np.nan] * len(weeks)
                for fid in features}
        max_weeks = max(max_weeks, len(weeks))
        cols['week'] = sorted(weeks)
        data[user] = pd.DataFrame(cols)
        data[user].set_index('week', inplace=True)

    print '%d users, %d weeks, %d features.' % (len(users), max_weeks,
                                                len(features))
    print 'done.'
    print 'setting all feature values...'
    pct = 5.0
    for i, row in enumerate(rows):
        complete = (float(i) / len(rows)) * 100.0
        if complete > pct:
            print '%.0f%% complete' % complete
            pct += 5.0
        user = row['user_id']
        week = int(row['longitudinal_feature_week'])
        feature_id = row['longitudinal_feature_id']
        value = row['longitudinal_feature_value']

        new_id = feature_id
        if feature_id in fmap:
            new_id = '%s_%s' % (fmap[feature_id], feature_id)

        data[user].ix[week, new_id] = value

    print 'done.'
    print 'saving data to csv...'

    try:
        os.makedirs(args.output_path + '/user_data/')
    except:
        pass

    for u, df in data.iteritems():
        with open('%s/user_data/%s.csv' % (args.output_path, u), 'w') as ufile:
            df.to_csv(ufile)
    print 'done.'

if __name__ == '__main__':
    main()
