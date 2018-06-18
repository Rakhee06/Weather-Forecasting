#!/usr/bin/python
"""SAOLA Algorithm. This is a mockup, include more methods if needed"""
<<<<<<< HEAD
import numpy as np
from scipy.stats.stats import pearsonr
=======
import os
import time
import pickle
import signal
import argparse
from multiprocessing import Pool, TimeoutError
import numpy as np
#from numpy_data import Data
from pickle_data_2 import Data

def comp_spearman(feature, label):
    """Use spearman correlation for feature against sum15"""
    return abs(feature.corr(label, method='spearman'))

def comp_pearson(var_a, var_b):
    """Use pearson correlation for this comparison"""
    return abs(var_a.corr(var_b))

def comp_conditioned(var_x, var_y, var_z):
    """Calculate partial correlation conditioned on labels
    Naive implementation:
    https://en.wikipedia.org/wiki/Partial_correlation#Using_recursive_formula
    case when Z is just one variable
    """
    rhoxy = var_x.corr(var_y)
    rhoxz = var_x.corr(var_z)
    rhozy = var_z.corr(var_y)

    return abs(rhoxy - rhoxz*rhozy)/(np.sqrt(1 - rhoxz**2)*np.sqrt(1 - rhozy**2))
>>>>>>> 05e11c3b88b3fb5313f29e74125ab6fdd8fffd84


def saola_algorithm(features, params, job=0, njobs=None):
    """The algorithm. I should return a list of tuples. Each tuple represents a
    single feature column.

    i.e.
    return [(4, 'pw', 4567), (3, 't850', 3456), ... , (2, 'z1000', 2345)]
    """

    if njobs is None:
        njobs = params['njobs']

<<<<<<< HEAD
if __name__ == "__main__":
    selected_features = run_saola()
    print selected_features
=======
    data = Data()

    #label_col = data.get_label_col()
    label_col = data.get_sum15_col()

    # Holds the features that are related to the label
    # key = tuple like (4, 'pw', 4567)
    # val['zscore_w_label'] = float, is z-score for feature against label
    # val['pd_series'] = pandas.Series, the actual column
    feature_dict = {}

    # variables for progress indicator
    total_count = 0.0
    max_count = len(features)
    display_progress = 0

    delta_1 = params['delta_1']
    delta_2 = params['delta_2']
    comp_function = comp_pearson

    prefix = "\033[F" * (njobs - job)
    suffix = "\n" * (njobs - job - 1)

    for col_id in features:

        # update and display progress indicator
        total_count += 1
        if njobs == 1 or total_count > display_progress:
            if njobs > 1:
                display_progress += 99
            percent = ((total_count/max_count)*100)
            print prefix + ("Job # %d: Working on %20s, num of features %3d: [% 5.1f%%]"
                            % (job, col_id, len(feature_dict), percent)) + suffix

        # this is the feature under consideration
        feature_i = data.get_feature_col(col_id)

        # relation with feature_i to label (z-score)
        feature_rel_label = comp_function(feature_i, label_col)
        if feature_rel_label < delta_1:
            # ignore this one, go to next feature
            continue
        else:
            rel_indicator = True

        # compare related feature to the other related features
        for key, val in feature_dict.items():
            # get column for feature_y
            feature_y = val['pd_series']
            # relation between feature_i to feature_y
            feat_i_rel_feat_y = comp_function(feature_i, feature_y)
            # taken from SAOLA paper page 663
            #delta_2 = min(feature_rel_label, val['zscore_w_label'])
            if (val['zscore_w_label'] > feature_rel_label
                    and feat_i_rel_feat_y >= delta_2):
                # DO NOT ADD feature_i to set of features
                rel_indicator = False
                break
            if (feature_rel_label > val['zscore_w_label']
                    and feat_i_rel_feat_y >= delta_2):
                # feature_i is better related to label than feature_y
                # delete feature_y
                feature_dict.pop(key)

        if rel_indicator:
            # add current feature to dict
            feature_dict[col_id] = {
                'zscore_w_label': feature_rel_label,
                'pd_series': feature_i}

    print prefix + ("Job # %d: Procesed %6d and selected %3d features: [% 5.1f%%]"
                    % (job, len(features), len(feature_dict), 100)) + " " * 40 + suffix

    output = feature_dict.keys()
    feature_dict = None
    data = None
    return output

def parallel_processing(features, params):
    """A method for processing any set of features, using multiprocessing"""
    njobs = params['njobs']
    nsplits = params['nsplits']
    group_size = int(np.ceil(float(len(features)) / njobs / nsplits))
    groups = [features[i:i + group_size] for i in xrange(0, len(features), group_size)]

    selected_features = []
    for iteration in range(nsplits):
        if nsplits > 1:
            print "\nIteration: %d/%d" % (iteration, nsplits - 1)
        print "\n" * njobs

        # Want workers to ignore Keyboard interrupt
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Create the pool of workers
        pool = Pool(njobs)

        # restore signals for the main process
        signal.signal(signal.SIGINT, original_sigint_handler)

        jobdata = {}
        partial_selections = {}

        try:
            for job in range(njobs):
                gnum = iteration * njobs + job
                jobdata[gnum] = pool.apply_async(saola_algorithm, (groups[gnum], params, job))

            pool.close()

            for job in range(njobs):
                gnum = iteration * njobs + job
                partial_selections[gnum] = jobdata[gnum].get(params['hard_limit'])
                selected_features += partial_selections[gnum]

        except KeyboardInterrupt:
            pool.terminate()
            print "\n\nCaught KeyboardInterrupt, workers have been terminated\n"
            raise SystemExit

        except TimeoutError:
            pool.terminate()
            print "\n\nThe workers ran out of time. Terminating process.\n"
            raise SystemExit

        #pool.join()
        del pool
        # clean stdout a little bit
        print "\033[F" * (njobs + 1) + " " * 71
        print "\n" * (njobs - 1)

    print "\nMerging"
    try:
        print "\n"
        merged_output = saola_algorithm(selected_features, params, njobs=1)
    except KeyboardInterrupt:
        print "\n\nCaught KeyboardInterrupt, workers have been terminated\n"
        raise SystemExit

    return merged_output

def run_saola(params):
    """The algorithm. I should return a list of tuples. Each tuple represents a
    single feature column.

    i.e.
    return [(4, 'pw', 4567), (3, 't850', 3456), ... , (2, 'z1000', 2345)]
    """

    data = Data()

    variables = ['z300', 'v850', 'u300', 'z1000', 'u850', 'z500',
                 'pw', 'v300', 't850']

    selected_features = {}
    for var in variables:
        features = []
        for location in range(1, 5328 + 1):
            for minusd in range(0, data.history_window):
                col_id = (minusd, var, location)
                features.append(col_id)
        selected_features[var] = parallel_processing(features, params)

    feature_set = []
    for var in variables:
        feature_set += selected_features[var]

    params['delta_2'] = 0.80

    return saola_algorithm(feature_set, params, njobs=1)

def main(params):
    """The main method"""

    start = time.time()
    selected_features = run_saola(params)
    end = time.time()

    hours, remainder = divmod(end - start, 3600)
    minutes, seconds = divmod(remainder, 60)
    print "\nRunning time: %02d:%02d:%05.2f\n" % (hours, minutes, seconds)

    outfile = open(os.path.join('data', params['output_file']), 'wb')
    pickle.dump(selected_features, outfile)
    outfile.close()
    print selected_features
    return selected_features

# parameter configurations

parser = argparse.ArgumentParser(description='Feature selection with Saola.')
# Main arguments
parser.add_argument('-d1', '--delta-1', dest='delta_1',
                    action='store', required=False, default=0.015, type=float,
                    help='Threshold for feature rejection')

parser.add_argument('-d2', '--delta-2', dest='delta_2',
                    action='store', required=False, default=0.015, type=float,
                    help='Threshold for feature redundancy check')

parser.add_argument('-rf', '--results-file', dest='output_file',
                    type=str, default='selected_features.pickle')

# Multiprocessing jobs
parser.add_argument('-nj', '--n-jobs', dest='njobs', action='store',
                    required=False, default=4, type=int,
                    help='The number of CPUs to use')

parser.add_argument('-ns', '--n-splits', dest='nsplits', action='store',
                    required=False, default=1, type=int,
                    help='The number of times to split the whole process')

parser.add_argument('-l', '--hard-time-limit', dest='hard_limit', action='store',
                    required=False, default=86400, type=int,
                    help='Stops the simulation after this many seconds')

parser.add_argument('-q', '--quiet', '--no-progress', dest='progress',
                    action='store_false',
                    help='Supress progress output')

parser.add_argument('-s', '--silent', dest='silent', action='store_true',
                    help='Supress all output')

parameters = vars(parser.parse_args())
if __name__ == "__main__":
    main(parameters)
>>>>>>> 05e11c3b88b3fb5313f29e74125ab6fdd8fffd84
