import numpy as np
import time

def euclid_dist_init(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


def euclid_dist_np_2nd(p, q):
    return np.sqrt(np.sum(np.square(np.array(p)) + np.square(np.array(q))))


def euclid_dist(p, q):
    if len(p) != len(q):
        raise ValueError("p and q must have the same dimensions")
    sum_sq_diff = sum((pi - qi) ** 2 for pi, qi in zip(p, q))
    return sum_sq_diff ** 0.5



np.random.seed(42)
test_points = {"blue": np.random.rand(100000, 3), "red": np.random.rand(100000, 3)}

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, points):
        self.points = points
        self._normalize_points_optimized()
        self._normalize_points_non_vectorized()

    def _normalize_points_optimized(self):
        all_points = np.vstack([self.points[category] for category in self.points])
        self.means = np.mean(all_points, axis=0)
        self.stds = np.std(all_points, axis=0)
        all_points_normalized = (all_points - self.means) / self.stds
        start_idx = 0
        for category in self.points:
            end_idx = start_idx + len(self.points[category])
            self.points[category] = all_points_normalized[start_idx:end_idx]
            start_idx = end_idx

    def _normalize_points_non_vectorized(self):
        all_points = np.array([point for category in self.points for point in self.points[category]])
        self.means = np.mean(all_points, axis=0)
        self.stds = np.std(all_points, axis=0)
        for category in self.points:
            self.points[category] = [(np.array(point) - self.means) / self.stds for point in self.points[category]]

# Instantiate the classifier
clf = KNearestNeighbors()

# Measure time for vectorized normalization
start_time = time.time()
clf.fit(test_points)
print("Vectorized normalization time:", time.time() - start_time)

# Reset points for a fair comparison
test_points = {"blue": np.random.rand(100000, 3), "red": np.random.rand(100000, 3)}

# Measure time for non-vectorized normalization
clf._normalize_points = clf._normalize_points_non_vectorized
start_time = time.time()
clf.fit(test_points)
print("Non-vectorized normalization time:", time.time() - start_time)

a = (3, 0)
b = (0, 4)

print(f"euclid_dist_bad for {a}, {b}: ", euclid_dist_init(a, b))
print(f"euclid_dist_np for {a}, {b}: ", euclid_dist_np_2nd(a, b))
print(f"euclid_dist for {a}, {b}: ", euclid_dist(a, b))

a = (5, 0)
b = (2, 4)

print(f"euclid_dist_bad for {a}, {b}: ", euclid_dist_init(a, b))
print(f"euclid_dist_np for {a}, {b}: ", euclid_dist_np_2nd(a, b))
print(f"euclid_dist for {a}, {b}: ", euclid_dist(a, b))

a = 3
b = 4

print(f"euclid_dist_bad for {a}, {b}: ", euclid_dist_init(a, b))
print(f"euclid_dist_np for {a}, {b}: ", euclid_dist_np_2nd(a, b))
print(f"euclid_dist for {a}, {b}: ", euclid_dist(a, b))
