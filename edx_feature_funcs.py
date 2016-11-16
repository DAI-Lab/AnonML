import pandas as pd
import numpy as np
from collections import OrderedDict
import pdb


# Labeling function: returns True, False, or None.
def label_dropout(df, pred_time, lead_time):
    # find the first row after our time cutoff, and the last row before it.
    last_row = df[df['week'] <= pred_time - lead_time].iloc[-1]
    next_row = df[df['week'] >= pred_time].iloc[0]

    # if someone has already dropped out before our cutoff, ignore their data.
    if last_row['dropout_1'] == 0:
        return 'dropout', None

    return ('dropout', next_row['dropout_1'] == 0)

def ff_last_week(df):
    features = OrderedDict()
    for f in df.columns:
        if f not in ['dropout_1', 'week']:
            features[f] = df.iloc[-1][f]

    return features

def ff_average(df):
    features = OrderedDict()
    for f in df.columns:
        if f not in ['dropout_1', 'week']:
            features[f + '_average'] = np.mean(df[f])

    return features
