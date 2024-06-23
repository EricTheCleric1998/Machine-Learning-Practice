# Ethan's Machine Learning Practice Assignment #1
# The purpose is to practice with the concept of K-Closest Neighbors
# By identifying K number of points closest to a new point the algorithm that
# will identify whether the new point is categorized as a Tuna or Salmon based
# on the K Closest Neighbors to the point.

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# The current directory is the src directory of the project. Go up one level (..). Then look in the data directory.
path = r'..\data\fish_data.csv'

# To help with understanding, print the absolute path
print(os.path.abspath(path))

# This one line reads the whole data file into a 2D array of 32-bit floats (skipping the header row).
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

# The : specifies a whole column of the 2D array, so these 2 prints shows the three columns
print("Length:\n" + str(data[:, 0]))  # Length (cm)
print("Weight:\n" + str(data[:, 1]))  # Weight (kg)
print("Classification:\n" + str(data[:, 2]))  # Classification [0 is Skipjack Tuna, 1 is Atlantic Salmon]

fig3 = plt.figure(1)

# Set up a color table so that 0=Skipjack Tuna is blue and 1=Atlantic Salmon is salmon.
my_colortable = ['blue', 'salmon']

# Create the colormap
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_salmon", my_colortable)

x = data[:, 0]  # Not necessary to use this temp variable, but makes the scatter function clearer
y = data[:, 1]
color = data[:, 2].astype(int)

plt.scatter(x, y, c=color, cmap=cmap, marker="*", s=50)
plt.xlabel("Length (cm)")
plt.ylabel("Weight (kg)")
plt.grid(color='lightgray', linestyle='--', linewidth=1)
plt.show()

# This is a very fast and powerful way to index NumPy arrays
true_if_tuna = data[:, 2] == 0
print("true_if_tuna:\n" + str(true_if_tuna))
tuna_length = data[true_if_tuna, 0]
tuna_width = data[true_if_tuna, 1]
print("tuna_length:\n" + str(tuna_length))


# Find the distance between our new point and the current point in the for loop
# l for length and w for width to make it look cleaner
def find_distance(l1, w1, l2, w2):
    result = np.sqrt((l2 - l1) ** 2 + (w2 - w1) ** 2)
    # The following commented out line helps evaluate the math that this function does
    # print("sqrt[(" + str(l2) + " - " + str(l1) + ")^2 + (" + str(w2) + " - " + str(w1) + ")^2] = " + str(result))
    return result


# Find the common type for the data entry point through an array of K-Closest Neighbors
# ARG arr is the array K_Closest_Neighbors
def find_common(arr):
    print("These are the closest points to the new point\n" + str(arr))
    tuna_counter = 0
    salmon_counter = 0
    # Iterate over each value within arr and determine its type dependent on the stored data entry
    for i in range(len(arr)):
        if arr[i, 1] == 0:
            tuna_counter += 1
        if arr[i, 1] == 1:
            salmon_counter += 1

    if tuna_counter > salmon_counter:
        return 0  # Tuna
    elif salmon_counter > tuna_counter:
        return 1  # Salmon
    else:
        return -97  # Not a tuna or a salmon


# Find the closest point to the indicated point
# Will try to implement K-Closest Neighbor algorithm value to play with K number of objects
# ARG l is the length of the new point
# ARG w is the weight of the new point
# ARG k is the number of closest neighbors wanted (it's better if the k is odd)
def find_closest(l, w, k):
    if k <= 0:
        return None

    # Initialize temporary arrays to find K-Closest points
    distance_arr = np.zeros(len(data))
    type_arr = np.zeros(len(data))
    k_closest_arr = np.zeros([k, 2])

    for i in range(np.size(data, axis=0)):
        # Increment current value
        x_val = data[i, 0]
        y_val = data[i, 1]
        type_val = data[i, 2]

        # Initialize values for each "result" array
        distance_arr[i] = find_distance(float(x_val), float(y_val), l, w)
        type_arr[i] = type_val

    print("Distance of all points from the new point:\n" + str(distance_arr))
    for j in range(k):
        # Find the index of min value in distance_arr
        index = np.argmin(distance_arr)

        # Set k_closest_arr value to the min value found
        k_closest_arr[j, 0] = distance_arr[index]
        k_closest_arr[j, 1] = type_arr[index]

        # Change min value to max value possible to make it possible to find next min value
        distance_arr[index] = 2147483647  # Max int32 value

    return k_closest_arr


def determine(fish_type):
    if fish_type == 0:
        print("The point is a Skipjack Tuna")
    elif fish_type == 1:
        print("The point is an Atlantic Salmon")
    else:
        print("The point is likely not a Skipjack Tuna nor an Atlantic Salmon")


# Just some test values
fish1 = np.array([50.0, 4.0])
fish2 = np.array([67.0, 4.3])
fish3 = np.array([75.0, 6.0])
fish4 = np.array([85.0, 7.0])

# Test #1
print("\nFish #1 is " + str(fish1[0]) + "cm and " + str(fish1[1]) + "kg:")
test1 = find_common(find_closest(fish1[0], fish1[1], 3))
determine(test1)
# Test #2
print("\nFish #2 is " + str(fish2[0]) + "cm and " + str(fish2[1]) + "kg:")
test2 = find_common(find_closest(fish2[0], fish2[1], 3))
determine(test2)
# Test #3
print("\nFish #3 is " + str(fish3[0]) + "cm and " + str(fish3[1]) + "kg:")
test3 = find_common(find_closest(fish3[0], fish3[1], 3))
determine(test3)
# Test #4
print("\nFish #4 is " + str(fish4[0]) + "cm and " + str(fish4[1]) + "kg:")
test4 = find_common(find_closest(fish4[0], fish4[1], 3))
determine(test4)
# Test #5
print("\nSame as Fish #4 but except we try with 5 Closest Neighbors instead of 3")
print("Fish #4 is " + str(fish4[0]) + "cm and " + str(fish4[1]) + "kg:")
test5 = find_common(find_closest(fish4[0], fish4[1], 5))
determine(test5)