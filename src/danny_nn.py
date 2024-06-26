from danny import npClosest
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

phisArray = np.genfromtxt(r'..\data\PhiUSIIL_Phishing_URL_Dataset.csv', delimiter=',', skip_header=1,
                          dtype=np.float64, encoding='utf-8', usecols=(0, 2, -1))

#normalizing values


xTrain, xTest, yTrain, yTest = train_test_split(phisArray[:, :-1], phisArray[:, -1:],
                                                random_state=10, test_size=0.1)

print(f'xTrain shape: {np.shape(xTrain)}')
print(f'xTest shape: {np.shape(xTest)}')
print(f'yTrain shape: {np.shape(yTrain)}')
print(f'yTest shape: {np.shape(yTest)}')

neighbors = 5
print(f"Starting K-Nearest Neighbors with k={neighbors}")

numVals = np.size(xTest, axis=0)
vals = range(numVals)
correct = 0
for i in vals:
    x1 = xTest[i]
    y1 = yTest[i]
    if npClosest(neighbors, phisArray, x1) == y1:
        correct = correct + 1
pyPercent = (correct / numVals) * 100

#Normalizing values
mins = xTrain.min(axis=0)
maxs = xTrain.max(axis=0)
xTrain = xTrain-mins
xTrain = xTrain/(maxs-mins)
xTrain[np.isnan(xTrain)] = 0


mins = xTest.min(axis=0)
maxs = xTest.max(axis=0)
xTest = xTest-mins
xTest = xTest/(maxs-mins)
xTest[np.isnan(xTest)] = 0





print()
print(f"Accuracy using 2 columns(0,2) and {neighbors} neighbors: ")
print(f"-Numpy nearest neighbors: {correct} / {numVals} correct for {pyPercent:.2f}%")
print()


#Setup NN
testShape = [np.size(xTrain, axis=1)]
model = tf.keras.Sequential([tf.keras.Input(testShape),
                             tf.keras.layers.Dense(units=2, activation='relu'),
                             tf.keras.layers.Dense(units=1, activation='sigmoid')])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(xTrain, yTrain, epochs = 100)


model.evaluate(xTest, yTest)

#### tests with numpy normalization
# test = np.array([[1,2,3],[-1,2,3],[1,-3,3]])
# mins = test.min(axis=0)
# maxs = test.max(axis=0)
# result = test-mins
# result = result/((maxs-mins))
# result[np.isnan(result)] = 0
# print(test)
# print(mins)
# print(result)

