from danny import npClosest
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

phisArray = np.genfromtxt(r'..\data\PhiUSIIL_Phishing_URL_Dataset.csv', delimiter=',', skip_header=1,
                          dtype= np.float64, encoding='utf-8', usecols=(1,3))
