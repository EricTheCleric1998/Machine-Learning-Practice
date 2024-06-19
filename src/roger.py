import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import pylab as pl

# Save for going back to 3d
# points = {"blue": [[2,4,3], [1,3,5], [2,3,1], [3,2,3], [2,1,6]],
          # "red": [[5,6,5], [4,5,2], [4,6,1], [6,6,1], [5,4,6], [10,10,4]]}

points = {"blue": [[2,4], [1,3], [2,3], [3,2], [2,1]],
          "red": [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [3, 3] #4

def euclid_dist(p, q):
    return np.sqrt(np.sum(np.array(p) - np.array(q)) ** 2)

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points
        self._normalize_points()

    def _normalize_points(self):
        all_points = np.array([point for category in self.points for point in self.points[category]])
        self.means = np.mean(all_points, axis=0)
        self.stds = np.std(all_points, axis=0)

        for category in self.points:
            self.points[category] = [(np.array(point) - self.means) / self.stds for point in self.points[category]]

    def _normalize_new_point(self, new_point):
        return (np.array(new_point) - self.means) / self.stds

    def predict(self, new_point):
        new_point = self._normalize_new_point(new_point)
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = euclid_dist(point, new_point)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

clf = KNearestNeighbors()
clf.fit(points)
print(clf.predict(new_point))

# Visualize

# Set aside 3d for normalization testing
# fig = pl.figure()
# ax = fig.add_subplot(projection="3d")
ax = pl.subplot()
ax.grid(True, color="#323232")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

for point in points['blue']:
    # ax.scatter(point[0], point[1], point[2], color="#104DCA", s=30)
    original_point = np.array(point) * clf.stds + clf.means
    ax.scatter(original_point[0], original_point[1], color="#104DCA", s=30)

for point in points['red']:
    # ax.scatter(point[0], point[1], point[2], color="#FF0000", s=30)
    original_point = np.array(point) * clf.stds + clf.means
    ax.scatter(original_point[0], original_point[1], color="#FF0000", s=30)

new_class = clf.predict(new_point)
color = "#FF0000" if new_class == "red" else "#104DCA"
ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)

normalized_new_point = clf._normalize_new_point(new_point)
for point in points['blue']:
    #ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#104DCA", linestyle="--", linewidth=2)
    original_point = np.array(point) * clf.stds + clf.means
    ax.plot([new_point[0], original_point[0]], [new_point[1], point[1]], color="#104DCA", linestyle="--", linewidth=2)

for point in points['red']:
    #ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#FF0000", linestyle="--", linewidth=2)
    original_point = np.array(point) * clf.stds + clf.means
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#FF0000", linestyle="--", linewidth=2)


plt.show()