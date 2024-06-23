import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plot_results = False
three_dim = False

# TODO: Incorporate commandline arguments for testing, read data from .csv files

# Simple testing, 2 blues and a 1 red should be the closest 3 with near identical distances
# points = {"blue": np.array([[6.0, 7], [5, 6], [2, 4], [3, 5], [7, 7]]),
          # "red": np.array([[5.0, 3], [5, 4], [3, 3], [6, 4], [4, 2]])}
# plot_results = True
# new_point = [3.0, 4]

# Middle scale testing for more densely populated graphs
np.random.seed(42)
points = {"blue": np.array([[2.9, 2.0], [4.5, 3.1], [7.8, 6.0], [9.0, 8.0], [3.0, 2.5], [6.0, 4.5], [10.0, 8.0],
                            [5.4, 4.0], [12.0, 10.0], [3.9, 3.0], [6.3, 5.0], [7.5, 5.5], [8.4, 7.0], [4.5, 3.5],
                            [11.7, 10.0], [9.6, 8.0], [10.8, 9.0], [12.6, 10.5], [3.6, 3.0], [8.1, 6.0], [5.7, 4.0],
                            [6.6, 5.5], [7.2, 6.0], [3.0, 2.5], [4.8, 4.0], [9.3, 7.5], [5.1, 4.0], [6.9, 5.5],
                            [7.8, 6.0], [3.3, 2.5], [5.4, 4.5], [8.4, 7.0], [4.2, 3.5], [10.2, 8.5],  [6.0, 5.0],
                            [7.5, 6.0], [3.9, 3.0], [8.7, 7.0], [5.7, 4.5], [9.9, 8.0], [11.4, 9.5], [6.6, 5.5],
                            [4.8, 4.0], [3.6, 3.0], [7.8, 6.0], [6.3, 5.0], [10.8, 9.0], [5.1, 4.0], [4.5, 3.5]]),
          "red": np.array([[2.0, 3.0], [3.1, 4.5], [6.0, 7.8], [8.0, 9.0], [2.5, 3.0], [4.5, 6.0], [8.0, 10.0],
                           [4.0, 5.4], [10.0, 12.0], [3.0, 3.9], [5.0, 6.3], [5.5, 7.5], [7.0, 8.4], [3.5, 4.5],
                           [10.0, 11.7], [8.0, 9.6], [9.0, 10.8], [10.5, 12.6], [3.0, 3.6], [6.0, 8.1], [4.0, 5.7],
                           [5.5, 6.6], [6.0, 7.2], [2.5, 3.0], [4.0, 4.8], [7.5, 9.3], [4.0, 5.1], [5.5, 6.9],
                           [6.0, 7.8], [2.5, 3.3], [4.5, 5.4], [7.0, 8.4], [3.5, 4.2], [8.5, 10.2], [5.0, 6.0],
                           [6.0, 7.5], [3.0, 3.9], [7.0, 8.7], [4.5, 5.7], [8.0, 9.9], [9.5, 11.4], [5.5, 6.6],
                           [4.0, 4.8], [3.0, 3.6], [6.0, 7.8], [5.0, 6.3], [9.0, 10.8], [4.0, 5.1], [3.5, 4.5]])}
new_point = [5.7, 6.8]
plot_results = True

# Large scale testing for runtime difference, disables visualization
# np.random.seed(42)
# points = {"blue": np.random.uniform(0, 20, (100000, 2)), "red": np.random.uniform(0, 20, (100000, 2))}
# new_point = np.random.uniform(0, 20, (1, 2)).flatten()

# 3d testing
# points = {"blue": [[2.0, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          # "red": [[5.0, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6]]}
# plot_results = True
# three_dim = True
# new_point = [3.0, 3, 4]


def np_euclid_dist(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


def euclid_dist(p, q):
    if len(p) != len(q):
        raise ValueError("p and q must have the same dimensions")
    sum_sq_diff = sum((pi - qi) ** 2 for pi, qi in zip(p, q))
    return sum_sq_diff ** 0.5


# Method for initializing the graph of each subplot
def init_graph(ax):
    ax.grid(True, color="#323232")
    ax.set_facecolor("black")
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")


# Method for plotting original points with no connecting lines
def plot_points(ax):
    for category in points:
        for point in points[category]:
            if three_dim:
                ax.scatter(point[0], point[1], point[2], color="#104DCA" if category == "blue" else "#FF0000", s=30)
            else:
                ax.scatter(point[0], point[1], color="#104DCA" if category == "blue" else "#FF0000", s=30)

    if three_dim:
        ax.scatter(new_point[0], new_point[1], new_point[2], color="#DFFFDF", marker="*", s=100, zorder=100)
    else:
        ax.scatter(new_point[0], new_point[1], color="#DFFFDF", marker="*", s=100, zorder=100)


# Method for plotting normalized graphs with predictions
def plot_prediction(ax):
    norm_new_point = clf._normalize_new_point(new_point)
    color = "#CF0000" if prediction == "red" else "#001D9A"
    if three_dim:
        ax.scatter(norm_new_point[0], norm_new_point[1], norm_new_point[2], color=color, marker="*", s=200, zorder=100)
    else:
        ax.scatter(norm_new_point[0], norm_new_point[1], color=color, marker="*", s=200, zorder=100)

    for category in points:
        for point in points[category]:
            color = "#104DCA" if category == "blue" else "#FF0000"
            if three_dim:
                ax.scatter(point[0], point[1], point[2], color=color, s=30)
                ax.plot([norm_new_point[0], point[0]], [norm_new_point[1], point[1]], [norm_new_point[2], point[2]],
                        color=color, linestyle="--", linewidth=1, alpha=0.6)
            else:
                ax.scatter(point[0], point[1], color=color, s=30)
                ax.plot([norm_new_point[0], point[0]], [norm_new_point[1], point[1]], color=color, linestyle="--",
                        linewidth=1, alpha=0.6)





# Modified to be further optimized, if
class KNearestNeighbors:
    def __init__(self, k=17):  # Adjust k for testing, 17 for middle, 73 for large
        self.k = k
        self.my_points = None
        self.means = None
        self.stds = None

    def _normalize_points(self):
        # vstack was a red herring, concatenate is better, though still has one loop
        all_points = np.concatenate([self.my_points[category] for category in self.my_points])

        # Mean and standard deviation calculation is the same
        self.means = np.mean(all_points, axis=0)
        self.stds = np.std(all_points, axis=0)

        # Normalize all points at once
        normalized_points = (all_points - self.means) / self.stds

        # Need categories for my data, but could be avoided by making category just another dimension of the vector,
        # as is done with the fish variables. If I could start over, would be the first thing I did this week.
        start_idx = 0
        for category in self.my_points:
            num_points = len(self.my_points[category])
            self.my_points[category] = normalized_points[start_idx:start_idx + num_points].tolist()
            start_idx += num_points

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

        # Prediction testing
        print("Method: predict")
        print(f"Normalized new point: {unkwn_pt}")
        print(f"Distances: {distances}")
        print(f"Selected categories: {categories}")

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

        # Prediction testing
        print("Method: bad_predict")
        print(f"Normalized new point: {unkwn_pt}")
        print(f"Distances (bad_predict): {distances}")
        print(f"Selected categories (bad_predict): {categories}")

        return result


clf = KNearestNeighbors()

# Normalization accuracy testing
# print("Before running")
# for category in points:
    # print(category, points[category])

# Visualize if told
# Modified to show 3 plots, actual, and both normalizations uses keyword for 3d version
subplot_key = {'projection': '3d'} if three_dim else {}
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=subplot_key)

if plot_results:
    # Actual values
    init_graph(axs[0])
    plot_points(axs[0])

# Timed for loops (bad)
start_time = time.time()
clf.bad_fit(points)
prediction = clf.bad_predict(new_point)
print(f"{prediction} predicted with loops in: {time.time() - start_time:.6f} seconds")

if plot_results:
    init_graph(axs[1])
    plot_prediction(axs[1])

# Further norm/revert testing
# print("After loop run, before reverting")
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
    init_graph(axs[2])
    plot_prediction(axs[2])

    plt.show()

# Final norm/revert testing
# print("After numpy run")
# for category in clf.my_points:
    # print(category, clf.my_points[category])
