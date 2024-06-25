import numpy as np
import math
import time
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
from sklearn.model_selection import train_test_split

#python closest neighbor function.
def kClosestN(k, testData, unknown):
    if k > len(testData):
        return "K is too large"
    else:
        distances = []
        for i in range(len(testData)):
            distances.append([])
            acc = 0
            for x in range(len(testData[i]) - 1):
                acc = acc + ((unknown[x] - testData[i][x]) ** 2)
            distances[i].append(math.sqrt(acc))
            type = testData[i][len(testData[i]) - 1]
            distances[i].append(type)
        sortedDist = sorted(distances, key=lambda dist: dist[0])
        types = {}

        # Count all types
        for i in range(k):
            key = sortedDist[i][1]
            if key in types:
                types[key] = types[key] + 1
            else:
                types[key] = 1
        # Find most common type in dictionary.
        num = 0
        mostCommon = None
        sortedTypes = dict(sorted(types.items()))
        for t in sortedTypes:
            if (sortedTypes[t] > num):
                mostCommon = t
                num = sortedTypes[t]
        return mostCommon


#numpy closest neighbor.
def npClosest(k, testData, unknown):
    if (k > np.size(testData, 0)):
        return "K is too large"
    else:
        preRoot = (testData[:, 0:-1] - unknown) ** 2
        distances = np.sqrt(np.reshape(np.sum(preRoot, axis=1), (-1, 1)))
        distType = np.c_[distances, testData[0:, -1:]]
        index = np.argpartition(distances, kth=k-1, axis=0)
        newArray = np.take_along_axis(distType, index[0:k, 0:], 0)
        typesFlat = newArray[:, 1:].flatten()
        values, counts = np.unique(typesFlat, return_counts=True)
        return values[counts.argmax()]


if __name__ == "__main__":
    # Command argument 1 of "data2D" gives a new 2d data set, "data3D" gives a new 3d set.
    if (len(sys.argv) > 1 and sys.argv[1] == "data2D"):
        testDataX, testDataY = make_classification(n_features=2, n_samples=100000, n_clusters_per_class=1,
                                                   n_informative=2,
                                                   n_redundant=0, scale=5, n_classes=4)
        testData = np.c_[testDataX, testDataY]
        np.savetxt(r'..\data\testData2D.csv', testData, delimiter=',')
    else:
        testData = np.genfromtxt(r'..\data\testData2D.csv', delimiter=',', dtype=np.float64)

    if (len(sys.argv) > 1 and sys.argv[1] == "data3D"):
        test2DataX, test2DataY = make_classification(n_features=3, n_samples=100000, n_clusters_per_class=1,
                                                     n_informative=2, n_redundant=0, scale=5, n_classes=4)
        testData2 = np.c_[test2DataX, test2DataY]
        np.savetxt(r'..\data\testData3D.csv', testData2, delimiter=',')
    else:
        testData2 = np.genfromtxt(r'..\data\testData3D.csv', delimiter=',', dtype=np.float64)

    path = r'..\data\fish_data.csv'
    data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

    path = r'..\data\3var_data.csv'
    data2 = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

    my_colortable = ['blue', 'salmon', 'black', 'purple']

    # Create the colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_salmon", my_colortable)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(testData2[:, 0], testData2[:, 1], testData2[:, 2], marker="o", c=testData2[:, 3], cmap=cmap)

    xTrain, xTest, yTrain, yTest = train_test_split(testData[:, :-1], testData[:, -1:],
                                                    test_size=0.05, random_state=10)

    xTrain2, xTest2, yTrain2, yTest2 = train_test_split(testData2[:, :-1], testData2[:, -1:],
                                                        test_size=0.05, random_state=10)


    start = time.time_ns()
    kClosestN(5, testData, [72.5, 5])
    end = time.time_ns()
    print(f"Python loop time (ns): {end - start}")


    start = time.time_ns()
    npClosest(5, testData, [72.5, 5])
    end = time.time_ns()
    print(f"NP function time (ns): {end - start}")

    print()
    print("Tests for csv file (1 is salmon, 0 is tuna) ")
    print("-Python results: ")
    print(kClosestN(5, data, [72.5, 5]))
    print(kClosestN(1, data, [67, 4.5]))
    print(kClosestN(8, data, [50, 10]))
    print("-Numpy Results: ")
    print(npClosest(5, data, [72.5, 5]))
    print(npClosest(1, data, [67, 4.5]))
    print(npClosest(8, data, [50, 10]))

    print()
    print("Tests for other data (4 classes): ")
    print(kClosestN(5, testData, [0, 0]))
    print(npClosest(5, testData, [0, 0]))

    print()
    print("Tests for 3 features: ")
    print(kClosestN(5, testData2, [0, 0, 0]))
    print(npClosest(5, testData2, [0, 0, 0]))

    print()
    print("3 value csv test, should be 0 for tuna: ")
    print(kClosestN(5, data2, [50, 3.1, 5]))
    print(npClosest(5, data2, [50, 3.1, 5]))

    # array tests
    # d = np.array([[1,2,3],
    #               [4,5,6]])
    # sub = [1,2,3]
    # print(d-sub)
    # print(np.reshape(np.array([1,2,3,4]), (-1, 1)))

    # Using test split for 2d accuracy tests.
    neighbors = 5
    numVals = np.size(xTest, axis=0)
    vals = range(numVals)
    correct = 0
    for i in vals:
        x1 = xTest[i]
        y1 = yTest[i]
        if npClosest(neighbors, testData, x1) == y1:
            correct = correct + 1
    pyPercent = (correct / numVals) * 100

    print()
    print(f"Accuracy for 2d four classes with {neighbors} neighbors: ")
    print(f"-Numpy nearest neighbors: {correct} / {numVals} correct for {pyPercent:.2f}%")
    print()

    # Tests for 3d accuracy.
    numVals = np.size(xTest2, axis=0)
    vals = range(numVals)
    correct = 0
    for i in vals:
        x1 = xTest2[i]
        y1 = yTest2[i]
        if npClosest(neighbors, testData2, x1) == y1:
            correct = correct + 1
    pyPercent = (correct / numVals) * 100

    print(f"Accuracy for 3d four classes with {neighbors} neighbors: ")
    print(f"-Numpy nearest neighbors: {correct} / {numVals} correct for {pyPercent:.2f}%")

    plt.show()
