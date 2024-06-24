This project has a .gitignore which should filterout from being pushed pycharm configuration files and other files that usually are not to be included in a repo. 

Put all code (.py files) in the src directory.

Put all data files (.csv, .png, .jpg, etc) in the data directory. If you have large groups of files, use subdirectories to orginize


##PhiUSIIL Phishing URL Dataset
The file data/PhiUSIIL_Phishing_URL_Dataset.csv is 
a substantial dataset comprising 134,850 legitimate 
and 100,945 phishing URLs. There are 51 features and
the label which is a binary classification: 1=legitimate,
0=phishing. Write and compare both a K-Nearest Neighbor
and a fully connected feed-forward neural network
(using TensorFlow) to classify and test your classifier.

Use the train_test_split function (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
to split the data into a 90% training segment and a 10% 
testing segment. Be sure to set random_state to get a 
repeatable split.

You do not need to use all 51 features. In machine learning, 
too many features can counter-productive. I suggest starting 
with the k-Nearest Neighbor classifier and picking 2 or 3 
features you think are most important. Next try adding, 
removing and replacing features. Do this until you 
find the feature with the best results. Work together on this 
with different of you trying different feature sets.

Try normalizing each column to a range from 0 to 1 by 
subtracting the minimum value
in each column from all values in the column, then dividing all values by the range
(maximum - minimum). There are other ways to normalize, which 
you are welcome to try, but this way is simple and generally 
effective.

Once you have found a good feature set for k-Nearest Neighbor 
classifier, add a second classifier: a fully connected, 
feed-forward neural network on the same set of features. 
Work on adjusting the NNs hyperparameters until you have it 
working at least as good as the k-Nearest Neighbor classifier.
The two primary hyperparameters to adjust are the number of
hidden layers, the number of nodes in each hidden layer.

Other hyperparameters are:
1) Activation Function: For a binary classifier, the most commonly used activation function **for the output layer** is the sigmoid function. For all other layers, ReLU or Leaky ReLU are usually best.
2) Optimizer: I suggest Adam
3) Weight and Biases initialization: I suggest starting with TensorFlow default (the biases are set to zero, while the weights are set according to the Glorot uniform initializer)

After you get working k-Nearest Neighbor and NN classifiers, fine tune the NN and compare results using
Monte Carlo Cross-Validation with 10 repeated random samplings.

