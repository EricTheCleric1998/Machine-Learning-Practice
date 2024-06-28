import os
import time
import roger
from sklearn.model_selection import train_test_split


def run_script():
    # Necessary variables, adjust neighbors as desired
    data_path = "../data/PhiUSIIL_Phishing_URL_Dataset.csv"
    neighbors = 7
    categories = 2
    # Columns correspond to URLLength, DomainLength, NoOfObfuscatedChar, and IsHTTPS
    columns = [0, 2, 12, 23]
    correct_predictions = 0

    if not os.path.isfile(data_path):
        print(f"Error: The provided file, '{data_path}', does not exist or could not be found")
        exit(1)

    # Read data in, add the final column to "columns" then parse down to only those columns
    data = roger.np.genfromtxt(data_path, delimiter=',', dtype=str, skip_header=1)
    columns.append(len(data[0]) - 1)
    relevant_data = data[:, columns].astype(roger.np.float32)

    # Separate points and labels
    raw_points = relevant_data[:, :-1]
    labels = relevant_data[:, -1]

    # Separate relevant data into two matching sets, the points and the labels
    url_train, url_test, labels_train, labels_test = train_test_split(raw_points, labels,
                                                                      test_size=0.1, random_state=12)
    # From a handful of native colors, pick one for each category
    category_labels = ["blue", "red", "cyan", "magenta", "orange", "maroon", "aqua"]
    points = {color: [] for color in category_labels[:categories]}

    # Set the data to match template
    for i in range(len(url_train)):
        category_index = int(labels_train[i])
        if category_index < len(category_labels):
            color = category_labels[category_index]
            points[color].append(url_train[i])

    # Convert lists to numpy arrays
    for color in points:
        points[color] = roger.np.array(points[color])

    # Initialize a KNearestNeighbors and fit the points
    clf = roger.KNearestNeighbors(neighbors)
    clf.fit(points)

    # Timer for the entire loop
    loop_time = time.time()
    for i in range(len(url_test)):
        # Keep separate so the label isn't normalized
        new_point = url_test[i]
        new_label = labels_test[i]
        # Timer for this prediction
        start_time = time.time()
        prediction, confidence = clf.predict(new_point)
        print(f"{prediction} predicted with numpy in {time.time() - start_time:.4f} seconds with "
              f"{confidence * 100:.2f}% confidence")

        if prediction == category_labels[int(new_label)]:
            correct_predictions += 1
        else:
            print(f"Inaccurate prediction! Current accuracy progress: {correct_predictions}/{len(url_test)}")

    # Cleanup, final overview
    final_time = time.time() - loop_time
    minutes = int(final_time // 60)
    seconds = final_time % 60
    print(f"Correct predictions: {correct_predictions}, {(correct_predictions / len(url_test)) * 100:.2f}% accurate")
    print(f"Approximate time: {minutes}:{seconds:.2f}")


if __name__ == "__main__":
    run_script()
