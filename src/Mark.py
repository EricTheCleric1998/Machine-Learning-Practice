import numpy as np
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import datetime
from collections import Counter

path = r'fish_data.csv'

# This one line reads the whole data file into a 2D array of 32-bit floats (skipping the header row).
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)
new_point = [60, 5]
points = data[:, :2]
types = data[1:, 2]

# Calculates the euclidean distance between two fish
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class kNN:
    def __init__(self, k):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points
        self.types = types


    # I stole the idea of zipping the values from Killian, it ended up being
    # super helpful as I was struggling with associating the type and distance
    # values.
    def predict(self, new_point):
        distances = []
        for point, type in zip(self.points, self.types):
            distances.append((euclidean_distance(point, new_point), type))

        categories = [category[1] for category in sorted(distances)[:self.k]]
        new_fish_type = Counter(categories).most_common(1)[0][0]

        if (new_fish_type == 1):
            print("Atlantic Salmon")
            return "Atlantic Salmon"

        else:
            print("Skipback Tuna")
            return "Skipback Tuna"

start = datetime.datetime.now()
clf = kNN(3)
clf.fit(points)
clf.predict(new_point)
end = datetime.datetime.now()
print("Completed in: " + str(end-start))
