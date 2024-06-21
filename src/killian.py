import math
import datetime
import numpy as np
from collections import Counter


# Method to find the Euclidean Distance between to points p1 and p2
# def euclidean_distance(p1, p2):
#     s = 0
#     for i in range(len(p1)):
#         s += (p1[i] - p2[i]) ** 2
#     return math.sqrt(s)

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


#  Path to csv to read data from
path = r'fish_data.csv'

# This one line reads the whole data file into a 2D array of 32-bit floats (skipping the header row).
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

points = data[:, :2]
labels = data[:, 2]


test_point = [73,5]


class KNearestNeighbors:
    # Constructor
    def __init__(self, k):
        self.k = k

    # Training data
    def fit(self, points):
        for i in data[0]:
            self.points = points
            self.labels = labels

    def predict(self, test_point):
        distances = []

        for point, label in zip(self.points, self.labels):
            distance = euclidean_distance(point, test_point)
            distances.append((distance, label))

        # distances.sort(key=lambda x: x[0])
        # neighbors = distances[:self.k]
        categories = [category[1] for category in sorted(distances)[:self.k]]

        result = Counter(categories).most_common(1)[0][0]

        fish_type = "Skipback Tuna" if result == 0 else "Atlantic Salmon"
        return fish_type


start = datetime.datetime.now()
clf = KNearestNeighbors(3)
clf.fit(points)
print(clf.predict(test_point))

end = datetime.datetime.now()
print(f"Execution time: {end - start}")