import roger
from sklearn.model_selection import train_test_split

data_path = "../data/Phishing_URL_Dataset.csv"
categories = 2
columns = [0, 2]

if not os.path.isfile(data_path):
    print(f"Error: The provided file, '{data_path}' does not exist or could not be found")

print(roger.os.path.abspath(data_path))
data = roger.np.genfromtxt(data_path, delimiter=',', dtype=np.float32, skip_header=1)
data = roger.np.delete(data, 1, axis=1)
columns.append(len(data) - 1)
relevant_data = data[:, columns]
url_train, url_test = train_test_split(relevant_data, test_size=0.1, random_state=12)
data_categories = url_train[:, -1]
category_labels = ["blue", "red", "cyan", "magenta", "orange", "maroon", "aqua"]
points = {color: [] for color in category_labels[:categories]}

for i in range(len(url_train)):
    category_index = int(data_categories[i])
    if category_index < len(category_labels):
        color = category_labels[category_index]
        points[color].append(url_train[i])

# Convert lists to numpy arrays
for color in points:
    points[color] = roger.np.array(points[color])

plot_results = False
three_dim = False
neighbors = 9999

clf = roger.KNearestNeighbors()

correct_predictions = 0
for point in url_test:
    new_point = url_test[point]
    clf.fit(points)
    prediction, confidence = clf.predict(new_point)
    print(f"{prediction} predicted with numpy in {time.time() - start_time:.6f} seconds with "
          f"{confidence * 100:.3f}% confidence")
    if prediction == category_labels[new_point[2]]:
        correct_predictions += 1

print(f"Correct predictions: {correct_predictions}, {(correct_predictions / len(url_test)) * 100}% accurate")