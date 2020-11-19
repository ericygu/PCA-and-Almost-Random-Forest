import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def standard_normalization(xTrain, xTest):
    standard = preprocessing.StandardScaler()
    standard_xTrain = standard.fit_transform(xTrain)
    standard_xTest = standard.transform(xTest)
    return standard_xTrain, standard_xTest


def pca(xTrain, xTest):
    pca_95 = PCA(.95)
    pca_xTrain = pca_95.fit_transform(xTrain)
    pca_xTest = pca_95.transform(xTest)
    return pca_xTrain, pca_xTest, pca_95.components_, pca_95.explained_variance_ratio_


def logistic_regression(xTrain, yTrain, xTest):
    lgr = LogisticRegression(penalty='none')
    lgr.fit(xTrain, yTrain.ravel())
    probability = lgr.predict_proba(xTest)[:, 1]
    return probability


def file_to_numpy(filename):
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    normalized_xTrain, normalized_xTest = standard_normalization(xTrain, xTest)
    pca_xTrain, pca_xTest, components, variance = pca(normalized_xTrain, normalized_xTest)
    normalized_probability = logistic_regression(normalized_xTrain, yTrain, normalized_xTest)
    pca_probability = logistic_regression(pca_xTrain, yTrain, pca_xTest)

    normalized_false_positive_rate, normalized_true_positive_rate, threshold = roc_curve(yTest, normalized_probability)
    pca_false_positive_rate, pca_true_positive_rate, threshold = roc_curve(yTest, pca_probability)

    plt.style.use('seaborn-dark')
    plt.figure(figsize=(15, 12))
    plt.plot(normalized_false_positive_rate, normalized_true_positive_rate, label='Normalized Dataset')
    plt.plot(pca_false_positive_rate, pca_true_positive_rate, label='PCA Dataset')
    plt.legend()


if __name__ == "__main__":
    main()
