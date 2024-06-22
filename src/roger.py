import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pylab as pl

# Save for going back to 3d
# points = {"blue": [[2,4,3], [1,3,5], [2,3,1], [3,2,3], [2,1,6]],
          # "red": [[5,6,5], [4,5,2], [4,6,1], [6,6,1], [5,4,6], [10,10,4]]}

# Simple testing, 2 blues and a 1 red should be the closest 3 with near identical distances
# points = {"blue": np.array([[2.0, 4], [1, 3], [2, 3], [3, 2], [2, 1]]),
          # "red": np.array([[5.0, 6], [4, 5], [4, 3], [6, 6], [5, 4]])}
# plot_results = True
# new_point = [3, 3] # 4

# Large scale testing for runtime difference
np.random.seed(42)
points = {"blue": np.random.uniform(0, 20, (100000, 3)), "red": np.random.uniform(0, 20, (100000, 3))}
new_point = np.random.uniform(0, 20, (1, 3))
plot_results = False

def np_euclid_dist(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


def euclid_dist(p, q):
    if len(p) != len(q):
        raise ValueError("p and q must have the same dimensions")
    sum_sq_diff = sum((pi - qi) ** 2 for pi, qi in zip(p, q))
    return sum_sq_diff ** 0.5

# Modified to be further optimized
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.my_points = None
        self.means = None
        self.stds = None

    # Vectorization attempt, successful resolution resulted in worse performance than original
    # def _normalize_points(self):
        # Use vstack to concatenate all points

        # all_points = np.vstack([self.my_points[category] for category in self.my_points])
        # self.means = np.mean(all_points, axis=0)
        # self.stds = np.std(all_points, axis=0)

        # Use vectorization to modify each category as a whole

        # for category in self.my_points:
            # self.my_points[category] = [(self.my_points[category] - self.means) / self.stds]

    def _normalize_points(self):
        all_points = np.array([point for category in self.my_points for point in self.my_points[category]])
        self.means = np.mean(all_points, axis=0)
        self.stds = np.std(all_points, axis=0)

        for category in self.my_points:
            self.my_points[category] = [(np.array(point) - self.means) / self.stds for point in self.my_points[category]]

    def _bad_normalize_points(self):
        # Calculate mean and standard deviation for each axis using for loops
        all_points = [point for category in self.my_points for point in self.my_points[category]]
        num_points = len(all_points)
        dim = len(all_points[0])
        sum_dim = [0] * dim

        # Mean
        for point in all_points:
            for i in range(dim):
                sum_dim[i] += point[i]

        self.means = [sum_dim[i] / num_points for i in range(dim)]

        # Std deviation
        sum_sq_diff = [0] * dim
        for point in all_points:
            for i in range(dim):
                sum_sq_diff[i] += (point[i] - self.means[i]) ** 2

        self.stds = [np.sqrt(sum_sq_diff[i] / num_points) for i in range(dim)]

        # Normalize points
        for category in self.my_points:
            for i in range(len(self.my_points[category])):
                for j in range(dim):
                    self.my_points[category][i][j] = (self.my_points[category][i][j] - self.means[j]) / self.stds[j]
    def _denormalize_points(self):
        for category in self.my_points:
            for i in range(len(self.my_points[category])):
                for j in range(len(self.my_points[category][i])):
                    self.my_points[category][i][j] = self.my_points[category][i][j] * self.stds[j] + self.means[j]

    def _normalize_new_point(self, pt):
        return (np.array(pt) - self.means) / self.stds

    def fit(self, pts):
        self.my_points = pts
        self._normalize_points()

    def bad_fit(self, pts):
        self.my_points = pts
        self._bad_normalize_points()


    def predict(self, pt):
        unkwn_pt = self._normalize_new_point(pt)
        distances = []

        # Vectorize distance calculation
        for category, points in self.my_points.items():
            dists = np.linalg.norm(points - unkwn_pt, axis=1)
            distances.extend(zip(dists, [category] * len(dists)))

        # Adjust result for vectorized distance calculation
        distances = np.array(distances)
        indices = np.argsort(distances[:, 0])[:self.k]
        categories = distances[indices, 1]
        result = Counter(categories).most_common(1)[0][0]

        # Prediction troubleshooting
        # print("Method: predict")
        # print(f"Normalized new point: {unkwn_pt}")
        # print(f"Distances: {distances}")
        # print(f"Selected categories: {categories}")

        return result

    def bad_predict(self, pt):
        unkwn_pt = self._normalize_new_point(pt)
        distances = []

        for category in self.my_points:
            for point in self.my_points[category]:
                distance = np_euclid_dist(point, unkwn_pt)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]

        # Prediction troubleshooting
        # print("Method: bad_predict")
        # print(f"Normalized new point: {unkwn_pt}")
        # print(f"Distances (bad_predict): {distances}")
        # print(f"Selected categories (bad_predict): {categories}")

        return result


clf = KNearestNeighbors()

# Normalization accuracy testing
# print("Before running")
# for category in points:
    # print(category, points[category])

# Timed for loops (bad)
start_time = time.time()
clf.bad_fit(points)
prediction = clf.bad_predict(new_point)
print(f"{prediction} predicted with loops in: {time.time() - start_time:.6f} seconds")

# Visualize
if plot_results:
    # Set aside 3d for normalization testing
    # fig = pl.figure()
    # ax = fig.add_subplot(projection="3d")

    # Modifyied to show 3 plots, actual, and both normalizations
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Actual values
    actual = axs[0]
    actual.grid(True, color="#323232")
    actual.set_facecolor("black")
    actual.figure.set_facecolor("#121212")
    actual.tick_params(axis="x", colors="white")
    actual.tick_params(axis="y", colors="white")

    # Modified to allow for any number of categories
    for category in points:
        for point in points[category]:
            original_point = np.array(point) * clf.stds + clf.means
            actual.scatter(original_point[0], original_point[1], color="#104DCA" if category == "blue" else "#FF0000",
                            s=30)

    actual.scatter(new_point[0], new_point[1], color="#DFFFDF", marker="*", s=100, zorder=100)

    loop_norm = axs[1]
    loop_norm.grid(True, color="#323232")
    loop_norm.set_facecolor("black")
    loop_norm.figure.set_facecolor("#121212")
    loop_norm.tick_params(axis="x", colors="white")
    loop_norm.tick_params(axis="y", colors="white")

    for category in points:
        for point in points[category]:
            loop_norm.scatter(point[0], point[1], color="#104DCA" if category == "blue" else "#FF0000", s=30)

    normalized_new_point = clf._normalize_new_point(new_point)
    color = "#FF0000" if prediction == "red" else "#104DCA"
    loop_norm.scatter(normalized_new_point[0], normalized_new_point[1], color=color, marker="*", s=200, zorder=100)

    for category in points:
        for point in points[category]:
            loop_norm.plot([normalized_new_point[0], point[0]], [normalized_new_point[1], point[1]],
                           color="#104DCA" if category == "blue" else "#FF0000", linestyle="--", linewidth=2)

# Further norm/revert testing
# print("After numpy run, before reverting")
# for category in clf.my_points:
    # print(category, clf.my_points[category])

# Revert points to original
clf._denormalize_points()

# print("After reverting")
# for category in clf.my_points:
    # print(category, clf.my_points[category])

# Timed numpy optimization
start_time = time.time()
clf.fit(points)
prediction = clf.predict(new_point)
print(f"{prediction} predicted with numpy in: {time.time() - start_time:.6f} seconds")

if plot_results:
    numpy_norm = axs[2]
    numpy_norm.grid(True, color="#323232")
    numpy_norm.set_facecolor("black")
    numpy_norm.figure.set_facecolor("#121212")
    numpy_norm.tick_params(axis="x", colors="white")
    numpy_norm.tick_params(axis="y", colors="white")

    for category in points:
        for point in points[category]:
            numpy_norm.scatter(point[0], point[1], color="#104DCA" if category == "blue" else "#FF0000", s=30)

    color = "#FF0000" if prediction == "red" else "#104DCA"
    numpy_norm.scatter(normalized_new_point[0], normalized_new_point[1], color=color, marker="*", s=200, zorder=100)

    for category in points:
        for point in points[category]:
            numpy_norm.plot([normalized_new_point[0], point[0]], [normalized_new_point[1], point[1]],
                            color="#104DCA" if category == "blue" else "#FF0000", linestyle="--", linewidth=2)

    # Final norm/revert testing
    # print("After loop run")
    # for category in clf.my_points:
        # print(category, clf.my_points[category])

    plt.show()

