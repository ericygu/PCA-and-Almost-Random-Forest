import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from random import sample


class RandomForest(object):
    nest = 0  # number of trees
    maxFeat = 0  # maximum number of features
    maxDepth = 0  # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None  # splitting criterion

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.forest = []

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        not_bs_list = []
        rows_range = range(0, len(xFeat))
        columns_range = range(0, len(pd.DataFrame(xFeat).columns))
        nest_range = range(1, 1 + self.nest)

        stats = dict.fromkeys(nest_range)

        for element in nest_range:
            samples = len(rows_range)
            indices, matrix, array = resample(rows_range, xFeat, y, n_samples=samples)
            features = sample(columns_range, self.maxFeat)

            tree = DecisionTreeClassifier(criterion=self.criterion,
                                          max_depth=self.maxDepth,
                                          min_samples_leaf=self.minLeafSample)

            matrix = matrix[:, features]
            tree.fit(matrix, array)
            self.forest.append({'tree': tree, 'features': features})

            not_bs_index = [i for i in rows_range if i not in indices]
            not_bs_list.append(not_bs_index)
            not_bs_predict = [m['tree'].predict(xFeat[:, m['features']]) for m in self.forest]
            not_bs_predict = np.array(not_bs_predict)

            yHatAlpha = np.zeros(len(xFeat))

            for i in range(len(yHatAlpha)):
                count = 0
                total = 0
                for j in range(len(self.forest)):
                    if i not in not_bs_list[j]:
                        continue
                    else:
                        count = count + 1
                        total = total + not_bs_predict[j][i]
                if count != 0:
                    if count < 2 * total:
                        continue
                    else:
                        yHatAlpha[i] = 1

            inaccuracy = 1 - accuracy_score(y, yHatAlpha)
            stats[element] = inaccuracy

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []

        prediction = [m['tree'].predict(xFeat[:, m['features']]) for m in self.forest]
        array_prediction = np.array(prediction)
        y = sum(array_prediction) / len(array_prediction)
        y_rounded = np.ndarray.round(y, decimals=0)
        yHat = y_rounded.astype(int)

        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    parser.add_argument("nest", type=int, help="amount of trees in the forest")
    parser.add_argument("maxFeat", type=int, help="maximum amount of features for every tree")
    parser.add_argument("criterion", type=str, help="type either 'entropy' or 'gini' for each valuation respectively")
    parser.add_argument("maxDepth", type=int, help="maximum depth of the tree")
    parser.add_argument("minLeafSample", type=int, help="the minimum samples of leaf nodes in the tree")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)
    model = RandomForest(args.nest, args.maxFeat, args.criterion, args.maxDepth, args.minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    print(accuracy_score(yTest, yHat))


if __name__ == "__main__":
    main()
