from killian import KNearestNeighbors
import numpy as np
import tensorflow as tf
import sklearn as sklearn

array = np.genfromtxt(r'PhiUSIIL_Phishing_URL_Dataset.csv', delimiter=',', skip_header=1,
                          dtype=np.float64, encoding='utf-8', usecols=(1, 3))

sklearn.model_selection.train_test_split(*array, test_size=.1, train_size=.9, random_state=42,
                                         shuffle=True, stratify=None)


def normalizing():
    return -1

