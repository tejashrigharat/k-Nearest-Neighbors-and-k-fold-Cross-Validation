# k-nearest neighbors on the hayes-roth Dataset
# References: https://machinelearningmastery.com/k-fold-cross-validation/
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# https://www.geeksforgeeks.org/minkowski-distance-python/
# https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
# https://www.geeksforgeeks.org/how-to-convert-pandas-dataframe-into-a-list/
# https://ljvmiranda921.github.io/notebook/2017/02/09/k-nearest-neighbors/
# https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

from random import seed
from random import randrange
import random as rn
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as hot
from math import *
from decimal import Decimal


# Loading a CSV file using pandas
def load_csv(filename):
    # dataset = list()
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.iloc[:, 1:]
    return dataset


# Find the minimum and maximum values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    # Shuffle the dataset
    shuffle = rn.sample(dataset, len(dataset))
    dataset_split = list()
    dataset_copy = list(shuffle)
    fold_size = int(len(shuffle) / n_folds)
    # print(len(dataset))

    # print("shuffle",shuffle)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    # print("fold",folds[1])
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    dist = np.sum(((np.array(row1[0:len(row1) - 1]) - np.array(row2[0:len(row2) - 1])) ** 2)) ** 0.5
    return dist


# Calculate the Manhattan distance between two vectors
def manhattan_distance(row1, row2):
    distance = 0.0
    dist = np.sum(abs(np.array(row1[0:len(row1) - 1]) - np.array(row2[0:len(row2) - 1])))
    return dist

# Calculate the Minkowski_distance distance between two vectors
def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) **
                 Decimal(root_value), 3)


def minkowski_distance(row1, row2, p_value):
    # pass the p_root function to calculate
    # all the value of vector parallely
    x = np.array(row1[0:len(row1) - 1])
    y = np.array(row2[0:len(row2) - 1])
    return (p_root(sum(pow(abs(a - b), p_value)
                       for a, b in zip(x, y)), p_value))

# Use oneHotEncoder to encode the nominal data
def encode(dataset):
    encoder = hot(handle_unknown='ignore')
    dt = pd.DataFrame(encoder.fit_transform(dataset.iloc[:, 0:len(dataset.iloc[0]) - 1]).toarray())
    encoder.get_feature_names()
    dt.columns = encoder.get_feature_names()
    dt["class"] = dataset.iloc[:, -1]
    dt = dt.values.tolist()
    return dt


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    if d == 1:
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
    elif d == 2:
        for train_row in train:
            dist = manhattan_distance(test_row, train_row)
            distances.append((train_row, dist))
    elif d == 3:
        for train_row in train:
            dist = minkowski_distance(test_row, train_row, p)
            distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)


# Test the kNN on the hayes-roth dataset
seed(1)
filename = 'hayes-roth.csv'
dataset = load_csv(filename)

dataset = encode(dataset)
# print(dataset)

# evaluate the entire algorithm
n_folds = 4
num_neighbors = 5
d = int(input("Enter value to select distance:\n1. Euclidean Distance\n2. Manhattan Distance\n3. Minkowski Distance\n"))

print('Hayes Roth Dataset')
if d == 1:
    print('Euclidean Distance is selected')
elif d == 2:
    print('Manhattan Distance is selected')
elif d == 3:
    print('Minkowski Distance is selected')
    p = int(input('Enter p-value to compute the Minkowski Distance\n'))

scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
