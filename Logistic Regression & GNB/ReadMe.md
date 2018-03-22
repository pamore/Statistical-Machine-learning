Implementation of Gaussian Naive Bayes and Logistic Regression.

Problem Statement:
Compare the two approaches on the bank note authentication dataset, which can be downloaded from http://archive.ics.uci.edu/ml/datasets/banknote+authentication. 

Implement a Gaussian Naive Bayes classifier(with conditional independent assumption) and a logistic regression classifier from scratch.

–For each algorithm: briefly describe how you implement it by giving the pseudocode.

–Plot a learning curve: the accuracy vs. the size of the training set. Plot 6 points for the curve, using [.01 .02 .05 .1 .625 1] RANDOM fractions of you training set and testing on the full test set each time. Average your results over 5 runs using each random fraction (e.g. 0.05) of the training set. Plot both the Naive Bayes and logistic regression learning curves on the same figure. For logistic regression, do not use any regularization term.

–Show the power of generative model: Use your trained Naive Bayes classifier (with the complete training set) to generate 400 examples from class y = 1. Report the mean and variance of the generated examples and the corresponding training data (for each fold, over 1 run). and compare with those in your training set (examples in training set with y = 1).
