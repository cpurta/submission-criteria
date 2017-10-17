# System
"""Concordance Checking."""
import logging
import os
import functools

# Third Party
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from bson.objectid import ObjectId

def has_concordance(validation_targets, submission_val, test_targets, submission_test, threshold=0.1):
    """Checks that the validation and test submission data have roughly equivalent log_losses based on

    Paramters:
    ----------
    validation_targets : ndarray
        the target values of the validation data in the competition

    submission_val : ndarray
        the submission probabilities that are used against the validation targets using the log_loss function

    test_targets : ndarray
        the target values of the test data in the competition

    submission_test : ndarray
        the submission probabilities that are used against the test targets using the log_loss function

    threshold : float, optional, default: 0.1
        The threshold in which our percent difference in log loss between validation and test data has to be under to have "concordance"

    Returns:
    --------
    concordance : bool
        Boolean value of the clustered submission data having concordance
    """
    if len(validation_targets) != len(submission_val):
        raise ValueError('submisison validation probabilities must be the same length as competition validation data')
    elif len(test_targets) != len(submission_test):
        raise ValueError('submisison test probabilities must be the same length as competition test data')

    val_log_loss = log_loss(validation_targets,submission_val)
    test_log_loss = log_loss(test_targets, submission_test)

    # score is the percent difference of the val_log_loss and test_log_loss
    score = 2.0 * (val_log_loss - test_log_loss) / (val_log_loss + test_log_loss)

    # if % difference is less than or equal to .1, call it good
    return score <= threshold

@functools.lru_cache(maxsize=2)
def get_competition_split(competition_id, db_manager, filemanager, sample_fraction=.25):
    """Return competition data that is split by the 'data_type' (validation, test)

    Parameters:
    -----------
    competition_id : int
        Numerical ID of the competition round of the tournament

    db_manager : DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager : FileManager
        S3 Bucket data access object for querying competition datasets

    Returns:
    --------
    validation_data : DataFrame
        all validation data from the tournament data for the current competition_id

    test_data : DataFrame
        all test data from the tournament data for the current competition_id

    """
    extract_dir = filemanager.download_dataset(competition_id)

    training = pd.read_csv(os.path.join(extract_dir, "numerai_training_data.csv"))
    tournament = pd.read_csv(os.path.join(extract_dir, "numerai_tournament_data.csv"))

    validation = tournament_data[tournament_data.data_type == 'validation']
    test = tournament_data[tournament_data.data_type == 'test']

    return validation, test.sample(frac=sample_fraction,axis='row')


def get_submission_split(submission_id, competition_id, validation_ids, test_ids, db_manager, filemanager):
    """Get validation and test data for a submission based on the submission_id

    Parameters:
    -----------
    submission_id : string
        ID of the submission

    competition_id : int
        Numerical ID of the competition round of the tournament

    validation_ids : ndarray
        array of the ids belonging to competition validation data

    test_ids : ndarray
        array of the ids belonging to competition test data

    db_manager : DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager : FileManager
        S3 Bucket data access object for querying competition datasets

    Returns:
    --------
    val_sub : DataFrame
        validation data from submission data

    test_sub : DataFrame
        test data from submission data

    """
    s3_file = db_manager.get_filename(submission_id)
    local_file = filemanager.download([s3_file])[0]
    data = pd.read_csv(local_file)

    val_sub = data[data.id.isin(validation_ids)]
    test_sub = data[data.id.isin(test_ids)]

    return val_sub, test_sub

def submission_concordance(submission, db_manager, filemanager):
    """Determine if a submission is concordant and write the result to MongoDB

    Parameters:
    -----------
    submission : dictionary
        Submission data that holds the ids of submission and competition round

    db_manager : DatabaseManager
        MongoDB data access object that has read and write functions to NoSQL DB

    filemanager : FileManager
            S3 Bucket data access object for querying competition datasets
    """
    s = db_manager.db.submissions.find_one({'_id':ObjectId(submission_id)})
    submission['user'] = s['username']
    submission['competition_id'] = s['competition_id']

    submission_id = submission['submission_id']
    competition_id = submission['competition_id']

    # tournament data
    validation, test = get_competition_split(competition_id, db_manager, filemanager)

    validation_ids = validation.id.values
    test_ids = test.id.values

    # submission data
    sub_val, sub_test = get_submission_split(submission_id, competition_id, validation_ids, test_ids, db_manager, file_manager)

    try:
        concordance = has_concordance(validation.target.values, sub_val.probability.values, test.target.values, sub_test.probability.values)
    except IndexError, ValueError:
        # If we had an indexing error, that is because the round restart, and we need to try getting the new competition variables.
         get_competition_split.cache_clear()
         validation, test = get_competition_split(competition_id, db_manager, filemanager)
         concordance = has_concordance(validation.target.values, sub_val.probability.values, test.target.values, sub_test.probability.values)

    db_manager.write_concordance(submsission_id, competition_id, concordance)
