import numpy as np
import math
import time
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
import matplotlib.pyplot as plt
import matplotlib.colors



path = r'..\data\fish_data.csv'
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)
testDataX, testDataY = make_classification(n_features=2, n_samples=100000, n_clusters_per_class=1, n_informative=2, n_redundant=0, scale=5, n_classes=4)
test2DataX, test2DataY = make_classification(n_features=3, n_samples=100000, n_clusters_per_class=1, n_informative=2, n_redundant=0, scale=5, n_classes=4)
testData = np.c_[testDataX, testDataY]
testData2 = np.c_[test2DataX, test2DataY]

path = r'..\data\3var_data.csv'
data2 = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

my_colortable = ['blue', 'salmon', 'black', 'purple']

# Create the colormap
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_salmon", my_colortable)

plt.figure(figsize=(8, 8))
plt.scatter(testDataX[:, 0], testDataX[:, 1], marker="o", c=testDataY, cmap=cmap)

#Python functions and loops.
def kClosestN(k, testData, unknown):
    if k > len(testData):
        return "K is too large"
    else:
        distances = []
        for i in range(len(testData)):
            distances.append([])
            acc = 0
            for x in range(len(testData[i])-1):
                acc = acc + ((unknown[x] - testData[i][x])**2)
            distances[i].append(math.sqrt(acc))
            type = testData[i][len(testData[i])-1]
            distances[i].append(type)
        sortedDist = sorted(distances, key=lambda dist: dist[0])
        types = {}

        #Count all types
        for i in range(k):
            key = sortedDist[i][1]
            if key in types:
                types[key] = types[key] + 1
            else:
                types[key] = 1
        #Find most common type in dictionary.
        num = 0
        mostCommon = None
        for t in types:
            if(types[t] > num):
                mostCommon = t
                num = types[t]
        return mostCommon



start = time.time_ns()
kClosestN(5, testData, [72.5, 5])
end = time.time_ns()
print("Python loop time (ns):")
print(end - start)


#Closest neighbors with numpy.
def npClosest(k, testData, unknown):
    if (k > np.size(testData, 0)):
        return "K is too large"
    else:
        preRoot = (testData[:, 0:-1] - unknown)**2
        distances = np.sqrt(np.reshape(np.sum(preRoot, axis=1), (-1, 1)))
        distType = np.c_[distances, testData[0:, -1:]]
        index = np.argpartition(distances, kth=0, axis=0)
        newArray = np.take_along_axis(distType, index[0:k, 0:], 0)
        values, counts = np.unique(newArray[:, 1:].flatten(), return_counts=True)
        return values[counts.argmax()]


start = time.time_ns()
npClosest(5, testData, [72.5, 5])
end = time.time_ns()
print("NP function time (ns): ")
print(end - start)

print()
print("Tests for csv file (1 is salmon, 0 is tuna): ")
print(kClosestN(5, data, [72.5, 5]))
print(npClosest(5, data, [72.5, 5]))
print(kClosestN(1, data, [67, 4.5]))
print(npClosest(1, data, [67, 4.5]))
print(kClosestN(8, data, [50,10]))
print(npClosest(8, data, [50, 10]))


print()
print("Tests for other data (4 classes): ")
print(kClosestN(5, testData, [0,0]))
print(npClosest(5, testData, [0, 0]))


print()
print("Tests for 3 features: ")
print(kClosestN(5, testData2, [0,0,0]))
print(npClosest(5, testData2, [0,0,0]))

print()
print("3 value csv test, should be 0 for tuna: ")
print(kClosestN(5, data2, [50,3.1,5]))
print(npClosest(5, data2, [50,3.1,5]))



#array tests
# d = np.array([[1,2,3],
#               [4,5,6]])
# sub = [1,2,3]
# print(d-sub)
#print(np.reshape(np.array([1,2,3,4]), (-1, 1)))


plt.show()




