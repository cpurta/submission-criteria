# System
"""Originality Checking."""
import logging
import math
import functools
import os
from threading import Lock

# Third Party
from scipy.stats import entropy
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial.distance import canberra
import numpy as np
import pandas as pd
from bson.objectid import ObjectId

lock = Lock()

@functools.lru_cache(maxsize=2048)
def get_submission(db_manager, filemanager, submission_id):
    """Gets the submission file from S3

    Parameters:
    -----------
    db_manager: DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager: FileManager
        S3 Bucket data access object for querying competition datasets

    submission_id : string
        The ID of the submission

    Returns:
    --------
    submission : ndarray
        2d array of the submission probabilities. First column is sorted by ID
        and second column is sorted by probability.
    """
    if not submission_id:
        return None

    s3_filename = db_manager.get_filename(submission_id)
    try:

        local_files = filemanager.download([s3_filename])
        if len(local_files) != 1:
            logging.getLogger().info("Error looking for submission {}, found files".format(submission_id, local_files))
            return None

        local_file = local_files[0]
    except Exception as e:
        logging.getLogger().info("Could not get submission {}".format(submission_id))
        return None

    df = pd.read_csv(local_file)

    df.sort_values("id", inplace=True)
    return df

def original(submission1, submission2, threshold=0.05):
    """Determines if two submissions are original

    Paramters:
    ----------
    submission1, submission2 : 1-D ndarrays
        Submission arrays that will be used in the Kolmogorov-Smirnov statistic
    threshold : float, optional, default: 0.05
        threshold in which the originality_score must be greater than to be "original"

    Returns:
    --------
    original : bool
        boolean value that indicates if a submission is original
    """
    score = originality_score(submission1, submission2)
    return score > threshold

def originality_score(data1, data2):
    """
    Computes the mean canberra distance between a user submission data and another submission

    Parameters
    ----------
    data1, data2 : ndarray
        Two arrays of sample observations assumed to be drawn from a
        continuous distribution. Arrays must be of the same size.

    Returns
    -------
    statistic : float
        mean canbera distance

    Raises:
    -------
    ValueError when data1 and data2 are not of equal length
    """

    n1 = data1.shape[0]
    n2 = data1.shape[0]

    if n1 != n2:
        raise ValueError("data1 and data2 must be the same shape")

    return np.mean((data1 - data2)**2)

def is_almost_unique(submission_data, submission, db_manager, filemanager, is_exact_dupe_thresh, is_similar_thresh, max_similar_models):
    """Determines how similar/exact a submission is to all other submission for the competition round

    Paramters:
    ----------
    submission_data : dictionary
        Submission metadata containing the submission_id and the user associated to the submission

    submission : DataFrame
        Submission data that contains the users submission

    db_manager : DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager : FileManager
        S3 Bucket data access object for querying competition datasets

    is_exact_dupe_thresh :
        Threshold for determining if a submission is and exact duplicate to another submission

    is_similar_thresh :
        Similarity threshold that determines if a submission is too similar and counts against the submissions originality

    max_similar_models :
        The max number of models that a submission is allow to be similar to

    Returns:
    --------
    bool
        Whether the submission data is considered to be original or not
    """
    # Get the tournament data
    extract_dir = filemanager.download_dataset(competition_id)
    tournament_data = pd.read_csv(os.path.join(extract_dir, "numerai_tournament_data.csv"))

    # split the tournament_data into the various types ('validation','test','live')
    tournament_data_types = [tournament_data[tournament_data.data_type == 'validation'],
    tournament_data[tournament_data.data_type == 'test'],
    tournament_data[tournament_data.data_type == 'live']]

    num_similar_models = 0
    similar_models = []

    date_created = db_manager.get_date_created(submission_data['submission_id'])

    for user_sub in db_manager.get_everyone_elses_recent_submssions(submission_data['competition_id'], submission_data['user'], date_created):
        with lock:
            other_submission = get_submission(db_manager, filemanager, user_sub["submission_id"])
        if other_submission is None:
            continue
        for data_type in tournament_data_types:
            submission_type = submission[submission.id.isin(data_type.id.values)].probability.values
            other_submission_type = other_submission[other_submission.id.isin(data_type.id.values)].probablility.values

            score = originality_score(submission_type, other_submission_type)

            is_not_a_constant = np.std(submission_type) > 0

            if is_not_a_constant and np.std(other_submission_type) > 0 :
                pearson_correlation = pearsonr(submission_type, other_submission_type)[0]
                spearman_correlation = spearmanr(submission_type, other_submission_type)[0]

                if np.abs(pearson_correlation) > 0.95:
                    logging.getLogger().info("Found a highly correlated (pearsonr) submission {} with score {}".format(user_sub["submission_id"], correlation))
                    return False

                if np.abs(spearman_correlation) > 0.95:
                    logging.getLogger().info("Found a highly correlated (spearmanr) submission {} with score {}".format(user_sub["submission_id"], correlation))
                    return False


            if score < is_exact_dupe_thresh:
                logging.getLogger().info("Found a duplicate submission {} with score {}".format(user_sub["submission_id"], score))
                return False

            if score <= is_similar_thresh:
                num_similar_models += 1
                similar_models.append(user_sub["submission_id"])
                if num_similar_models >= max_similar_models:
                    logging.getLogger().info("Found too many similar models. Similar models were {}".format(similar_models))
                    return False

    return True


def submission_originality(submission_data, db_manager, filemanager):
    """Pulls submission data from MongoDB and determines the originality score and will update the submissions originality score

    This checks a few things
        1. If the current submission is similar to the previous submission, we give it the same originality score
        2. Otherwise, we check that it is sufficently unique. To check this we see if it is A. Almost identitical to
        any other submission or B. Very similar to a handful of other models.

    Parameters:
    -----------
    submission_data : dictionary
        Metadata about the submission pulled from the queue

    db_manager : DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager : FileManager
        S3 Bucket data access object for querying competition datasets
    """
    s = db_manager.db.submissions.find_one({'_id':ObjectId(submission_data['submission_id'])})
    submission_data['user'] = s['username']
    submission_data['competition_id'] = s['competition_id']
    logging.getLogger().info("Scoring {} {}".format(submission_data['user'], submission_data['submission_id']))

    with lock:
        submission = get_submission(db_manager, filemanager, submission_data['submission_id'])

    if submission is None:
        logging.getLogger().info("Couldn't find {} {}".format(submission_data['user'], submission_data['submission_id']))
        return

    is_exact_dupe_thresh = 0.01
    is_similar_thresh = 0.1
    max_similar_models = 1

    is_original = is_almost_unique(submission_data, submission, db_manager, filemanager, is_exact_dupe_thresh, is_similar_thresh, max_similar_models)
    db_manager.write_originality(submission_data['submission_id'], submission_data['competition_id'], is_original)
