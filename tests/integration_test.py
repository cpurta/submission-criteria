#!/usr/bin/env python
"""Integration testing."""

# System
import os

# Third Party
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# First Party
from testing_api import NumerAPI


def main():
    email = ""
    password = ""

    napi = NumerAPI()
    napi.credentials = (email, password)

    test_csv = "test_csv"
    if not os.path.exists(test_csv):
        os.makedirs(test_csv)

    if not os.path.exists("numerai_datasets"):
        print("Downloading the current dataset...")
        os.makedirs("numerai_dataset")
        napi.download_current_dataset(dest_path='numerai_dataset', unzip=True)
    else:
        print("Found old data to use.")

    training_data = pd.read_csv('numerai_datasets/numerai_training_data.csv', header=0)
    tournament_data = pd.read_csv('numerai_datasets/numerai_tournament_data.csv', header=0)

    features = [f for f in list(training_data) if "feature" in f]
    X, Y = training_data[features], training_data["target"]

    x_prediction = tournament_data[features]
    ids = tournament_data["id"]

    valid = tournament_data["data_type"] == "validation"
    test = tournament_data["data_type"] != "validation"

    x_pv, ids_v = x_prediction[valid], ids[valid]
    x_pt, ids_t = x_prediction[test], ids[test]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100),
            KNeighborsClassifier(10, n_jobs=-1),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1, hidden_layer_sizes=(100, 100)),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            LogisticRegression(n_jobs=-1)]

    for clf in clfs:
        clf_str = str(clf).split("(")[0]
        print("Training a {}".format(clf_str))
        clf.fit(X, Y)

    for clf in clfs:
        y_prediction = clf.predict_proba(x_prediction)
        results = y_prediction[:, 1]
        results_df = pd.DataFrame(data={'prediction': results})
        joined = pd.DataFrame(ids).join(results_df)

        out = os.path.join(test_csv, "{}-legit.csv".format(clf_str))
        print("Writing predictions to {}".format(out))
        # Save the predictions out to a CSV file
        joined.to_csv(out, index=False)

        napi.upload_prediction(out)

        input("Both concordance and originality should pass. Press enter to continue...")

    for i, clf1 in enumerate(clfs):
        for j, clf2 in enumerate(clfs):
            if i == j:
                continue

            y_pv = clf1.predict_proba(x_pv)[:, 1]
            valid_df = pd.DataFrame(ids_v).join(pd.DataFrame(data={'prediction': y_pv}))

            y_pt = clf2.predict_proba(x_pt)[:, 1]
            test_df = pd.DataFrame(ids_t).join(pd.DataFrame(data={'prediction': y_pt}))

            mix = pd.concat([valid_df, test_df])

            out = os.path.join(test_csv, "{}-{}-mix.csv".format(str(clf1).split("(")[0], str(clf2).split("(")[0]))
            mix.to_csv(out, index=False)
            print("Writing predictions to {}".format(out))

            napi.upload_prediction(out)
            input("Concordance should fail. Press enter to continue...")


if __name__ == '__main__':
    main()
