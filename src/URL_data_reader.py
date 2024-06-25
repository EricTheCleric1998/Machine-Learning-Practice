import numpy as np
import os

path = r'../data/PhiUSIIL_Phishing_URL_Dataset.csv'

# To help with understanding, print the absolute path
print(os.path.abspath(path))

data = np.genfromtxt(path, delimiter=",", dtype=str)
print(data.shape)

header_row = data[0, :]
print(f'header_row = {header_row}')

# remove header
data = data[1:, :]
first_row = data[0, :]
print(f'first_row={first_row}')

# the label is in column 49. Therefore, the next line will put all the labels into a vector.
label = data[:, 49].astype(np.float64)
print(f'first 15 labels = {label[0:15]}')


