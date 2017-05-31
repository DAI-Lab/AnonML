import pandas as pd
import numpy as np
from collections import OrderedDict
import pdb

included_features = [
    # total time spent on all events in a week
    'sum_observed_events_duration_2',
    # number of different problems attempted
    'distinct_attempts_6',
    # number of actual submission attempts
    'number_of_attempts_7',
    # total number of problems correct
    'distinct_problems_correct_8',
    # attempts per problem, not per correct answer: feature 7 / feature 6
    'average_number_of_attempts_9',
    # feature 2 / feature 8
    'sum_observed_events_duration_per_correct_problem_10',
    # feature 7 / feaure 8
    'number_problem_attempted_per_correct_problem_11',
    # avg(max(attempt.timestamp) - min(attempt.timestamp))
    'average_time_to_solve_problem_12',
    # variance of a student's observed timestamps
    'observed_event_timestamp_variance_13',
    # longest observed event
    'max_duration_resources_15',
    # amount of time spent on "lecture" resources in a week
    'sum_observed_events_lecture_16',
    # amount of time spent on "wiki" resources in a week
    #'sum_observed_events_wiki_18',
    # total number of attempts that were correct (kinda stupid)
    'attempts_correct_208',
    # percent of total submissions that were correct
    'percent_correct_submissions_209',
    # average time before deadline of submission - not correct submission or
    'average_predeadline_submission_time_210',
    # standard deviation of hour of the day of events
    # i.e. how much does the time of day a user is active vary?
    'std_hours_working_301',
]

label_name = 'dropout'

# Labeling function: returns True, False, or None.
def label_dropout(df, pred_time, lead_time):
    # find the first row after our prediction time, and the last row before our
    # lead time.
    last_row = df[df['week'] <= pred_time - lead_time].iloc[-1]
    next_row = df[df['week'] >= pred_time].iloc[0]

    # if someone has already dropped out before our cutoff, ignore their data.
    if last_row['dropout_1'] == 0:
        return 'dropout', None

    return ('dropout', int(next_row['dropout_1'] == 0))

def ff_last_week(df):
    features = OrderedDict()
    for f in df.columns:
        if f in included_features:
            features[f] = df.iloc[-1][f]

    return features

def ff_week_over_week(df):
    features = OrderedDict()
    for f in df.columns:
        if f in included_features:
            features[f + '_difference'] = df.iloc[-1][f] - df.iloc[-2][f]

    return features

def ff_trend(df):
    features = OrderedDict()
    for f in df.columns:
        if f in included_features:
            features[f + '_trend'] = np.polyfit(range(len(df[f])),
                                                np.nan_to_num(df[f]), 1)[0]

    return features

def ff_average(df):
    features = OrderedDict()
    for f in df.columns:
        if f in included_features:
            features[f + '_average'] = np.mean(df[f][:-1])

    return features
