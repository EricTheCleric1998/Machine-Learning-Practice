import math
import datetime
import time

import numpy as np
import os
import random
import matplotlib
import matplotlib.pyplot as plt

path = r'fish_data.csv'

# This one line reads the whole data file into a 2D array of 32-bit floats (skipping the header row).
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)


## Implementation of Randomized Quick Sort

# Swaps two elements in an array
def swap(array, ind1, ind2):
    temp = array[ind1]
    array[ind1] = array[ind2]
    array[ind2] = temp
    return array


# Partitions an array based on a randomly selected pivot value
def partition(array, low, high):
    pivot = random.randint(low, high)
    swap(array, pivot, high)
    pivotIndex = low
    for i in range(low, high):
        if (array[i] <= array[high]):
            swap(array, pivotIndex, i)
            pivotIndex += 1
    swap(array, pivotIndex, high)
    return pivotIndex


# Recursively partitions and sorts an array
def rqs(array, low, high):
    if (low < high):
        p = partition(array, low, high)
        rqs(array, low, p - 1)
        rqs(array, p + 1, high)


# A 'fish' data type to hold the length, weight, and type values
# from the csv file.
class Fish:
    def __init__(self, length, weight, type):
        self.length = length
        self.weight = weight
        self.type = type
        self.distance = None

    # Overrides the equality functions to be based on euclid distance
    # from a test point.
    def __eq__(self, other):
        return self.distance == other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __rt__(self, other):
        return self.distance > other.distance


# Initializes the list of fish objects from the csv and creates a test point
fishList = []
testFish = Fish(60, 5, None)

for i in range(len(data)):
    fishList.append(Fish(data[i][0], data[i][1], data[i][2]))


# Calculates the euclidean distance between two points
def euclideanDistance(p, q):
    return math.sqrt(((p.length - q.length) ** 2) + ((p.weight - q.weight) ** 2))


# Calculates the euclidean distances between each fish and the
# selected fish and sorts the fish list based on these distances
def predict(k, new_point):
    atlCount = 0
    skpCount = 0

    # Finds all euclidean distances from new_point and updates these values
    for p in fishList:
        p.distance = euclideanDistance(p, new_point)

    # Sorts the fish list based off of their distance from new_point
    rqs(fishList, 0, len(fishList) - 1)

    # Checks the first k values of the sorted list, and counts the number
    # of tuna/salmon samples.
    for i in range(k):
        if (fishList[i].type == 1):
            atlCount += 1
        else:
            skpCount += 1

    # Depending on the value of the counters, sets the new_points type and prints
    # to the console
    if atlCount > skpCount:
        new_point.type = 1
        print("Atlantic Salmon")

    else:
        new_point.type = 0
        print("Skipjack Tuna")

# Runs the function and keeps track of time.
start = datetime.datetime.now()
predict(3, testFish)
end = datetime.datetime.now()
print("Completed in: " + str(end-start))
