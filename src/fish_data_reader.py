import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# NOTE: depending on how you set up your project environment, this MAY NOT be the correct path to fish_data.csv for you.
#       If when you run, fish_data.csv is not found, then try removing teh ..\ from the beginning of the path.
# The current directory is the src directory of the project. Go up one level (..). Then look in the data directory.
path = r'../data/fish_data.csv'

# To help with understanding, print the absolute path
print(os.path.abspath(path))

# This one line reads the whole data file into a 2D array of 32-bit floats (skipping the header row).
data = np.genfromtxt(path, delimiter=',', dtype=np.float32, skip_header=1)

# The : specifies a whole column of the 2D array, so these 2 prints shows the three columns
print(data[:, 0])
print(data[:, 1])
print(data[:, 2])

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
print(true_if_tuna)
tuna_length = data[true_if_tuna, 0]
tuna_width = data[true_if_tuna, 1]
print(tuna_length)
