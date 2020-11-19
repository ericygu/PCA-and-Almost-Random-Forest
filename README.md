# PCA-and-Almost-Random-Forest

For PCA, the file is labeled Q1. 
For this problem, we quantify the impact of dimensionality reduction on logistic regression.
(a) Normalized the features of the wine quality dataset (where applicable). Trained an
unregularized logistic regression model on the normalized dataset and predicted the
probabilities on the normalized test data.
(b) Rann PCA on the normalized training dataset. 
(c) Trained an unregularized logistic regression models using the PCA dataset and predict the
probabilities on the appropriately transformed test data (i.e., for PCA, the test data
should be transformed to reflect the loadings on the k principal components). 
Plotted the ROC curves for both models (normalized dataset, PCA dataset) on the same graph.

The "Almost Random Forest" is labeled rf.

Just simply implemented a variant of the random forest using the decision
trees from scikit-learn.  Instead of subsetting the features for each node of each
tree in your forest, chose a random subspace that the tree will be created on. 
Each tree is built using a bootstrap sample and random subset of the features.

(a) Built the adaptation of the random forest, supporting the following parameters:

• nest: the number of trees in the forest

• maxFeat: the maximum number of features to consider in each tree

• criterion: the split criterion – either gini or entropy

• maxDepth: the maximum depth of each tree

• minSamplesLeaf: the minimum number of samples per leaf node

• train: Given a feature matrix and the labels, learn the random forest using the
data. The return value should be the OOB error associated with the trees up to that
point. For example, at 5 trees, calculate the random forest predictor by averaging
only those trees where the bootstrap sample does not contain the observation.

• predict: Given a feature matrix, predict the responses for each sample.
