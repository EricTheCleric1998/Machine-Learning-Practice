import os
import time
import roger
from sklearn.model_selection import train_test_split


# TODO: Certain problem: Only runs with a valid command line argument, had the idea to use main in roger.py and
#       run_script as a fix, but having trouble figuring out where to put the main wrapper in roger.py as I need access
#       to the class. Potential alternate resolution, relocate most of this code to roger.py and run it when the
#       command line argument is "URLTest"
#       Tentative problems, code does run, but it seems to be finding every point as blue with 100% certainty, but
#       at least recognizes "red" values as inaccurate. Runs at about 3 predictions a second, might have k too high
def run_script():
    data_path = "../data/PhiUSIIL_Phishing_URL_Dataset.csv"
    categories = 2
    columns = [0, 2]

    if not os.path.isfile(data_path):
        print(f"Error: The provided file, '{data_path}' does not exist or could not be found")
        exit(1)

    print(os.path.abspath(data_path))
    data = roger.np.genfromtxt(data_path, delimiter=',', dtype=str, skip_header=1)
    columns.append(len(data[0]) - 1)
    relevant_data = data[:, columns].astype(roger.np.float32)

    # Separate points and labels
    raw_points = relevant_data[:, :-1]
    labels = relevant_data[:, -1]

    url_train, url_test, labels_train, labels_test = train_test_split(raw_points, labels, test_size=0.1, random_state=12)
    data_categories = url_train[:, -1]
    category_labels = ["blue", "red", "cyan", "magenta", "orange", "maroon", "aqua"]
    points = {color: [] for color in category_labels[:categories]}

    for i in range(len(url_train)):
        category_index = int(labels_train[i])
        if category_index < len(category_labels):
            color = category_labels[category_index]
            points[color].append(url_train[i])

    # Convert lists to numpy arrays
    for color in points:
        points[color] = roger.np.array(points[color])

    neighbors = 99

    clf = roger.KNearestNeighbors(neighbors)
    clf.fit(points)
    correct_predictions = 0

    for i in range(len(url_test)):
        new_point = url_test[i]
        new_label = labels_test[i]
        start_time = time.time()
        prediction, confidence = clf.predict(new_point)
        print(f"{prediction} predicted with numpy in {time.time() - start_time:.6f} seconds with "
              f"{confidence * 100:.3f}% confidence")
        if prediction == category_labels[int(new_label)]:
            correct_predictions += 1
        else:
            print(f"Innacurate prediction! Current accuracy progress: {correct_predictions}/{len(url_test)}")

    print(f"Correct predictions: {correct_predictions}, {(correct_predictions / len(url_test)) * 100:.2f}% accurate")


if __name__ == "__main__":
    run_script()

