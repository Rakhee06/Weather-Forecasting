<<<<<<< HEAD
"""Module docstring"""
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

=======
#!/usr/bin/python
"""Module docstring"""
import os
import numpy as np
from pickle_data_2 import Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

### classifier methods ###

def linear_discriminant_analysis(data):
    """Linear Discriminant Analysis"""
    clf = LinearDiscriminantAnalysis()
    clf.name = "LDA"
    train_predict_and_results(data, clf)

def nearest_neighbors_classifier(data):
    """K Nearest neighbors classification"""
    clf = KNeighborsClassifier(3, 'distance')
    clf.name = "KNN"
    train_predict_and_results(data, clf)

def support_vector_machine(data):
    """Support Vector Machines"""
    clf = SVC()
    clf.name = "SVC"
    train_predict_and_results(data, clf)

def gaussian_naive_bayes(data):
    """Naive Bayes"""
    clf = GaussianNB()
    clf.name = "GaussNB"
    train_predict_and_results(data, clf)

def logistic_regression(data):
    """Logistic Regression """
    clf = LogisticRegression()
    clf.name = "LoReg"
    train_predict_and_results(data, clf)

def random_forest(data):
    """Random Forest"""
    clf = RandomForestClassifier()
    clf.name = "RNDForest"
    train_predict_and_results(data, clf)

### End of classifier methods ###
>>>>>>> 05e11c3b88b3fb5313f29e74125ab6fdd8fffd84

def normalize(data):
    """Returns data with columns normalized
    input: numpy array
    output: numpy array
    """
    # normalize data and return
    # https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
    return (data - data.min(axis=0)) / data.ptp(axis=0)

<<<<<<< HEAD
def load_data():
    """Reads datafile and returns data as numpy array"""

    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.astype.html
    data = np.load("phase3-data/data_selected_1980_2010.npy").astype(float)

    return normalize(data)

def load_target(column="label"):
    """Reads target labels and returns two columns: sum15 and label"""

    columns = {"sum15": 0, "label": 1}
    if column not in columns.keys():
        raise ValueError("%s is not in target data" % column)

    filepath = os.path.join("phase3-data", "target_1980_2010.npy")
    target = np.load(filepath)

    # lets normalize, sum15 might need it
    target = normalize(target)

    # return correct column
    return target[:, columns[column]]

def concat_data(data, target):
    '''Merge dataframe data with dataframe target and returns the final one '''
    
    final_data = np.concatenate((data,target[:,None]), axis=1)
    
    return final_data
=======
def load_ta_data():
    """Reads datafile and returns data as numpy array"""
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.astype.html
    data = np.load("data/data_selected_1980_2010.npy").astype(float)
    return normalize(data)

def load_ta_target():
    """Reads target labels and returns two columns: sum15 and label"""
    filepath = os.path.join("data", "target_1980_2010.npy")
    target = np.load(filepath)
    return target[:, 1]

def load_own_data():
    """Loads data corresponding to selected features by custom saola algorithm"""
    data = Data()
    features = data.read_selected_features()
    dataframe = data.get_dataframe_with(features)
    return normalize(dataframe.values)

def load_own_target():
    """Loads target column as stored in our data files"""
    data = Data()
    target = data.get_label_col()
    return target.values
>>>>>>> 05e11c3b88b3fb5313f29e74125ab6fdd8fffd84

def split_samples(data):
    """Splits data into training samples and test samples
    input: numpy array

    returns tuple (training_samples, test_samples)
    both are numpy arrays
    """

    training_samples = data[0:9497]
    test_samples = data[9497:11300]

    return training_samples, test_samples

<<<<<<< HEAD
def main():
    """The main method"""

    feat_data = load_data()
    label_data = load_target()
    #final = concat_data(feat_data, label_data)

    #print final
    X_training, X_test = split_samples(feat_data)
    Y_training, Y_test = split_samples(label_data)

    #10- fold cross-validation
    #knn = KNeighborsClassifier(n_neighbors=3)
    lda = LinearDiscriminantAnalysis(n_components=3, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
    #folds = cross_val_score(lda, X_training, Y_training, cv=10)
    #print folds

    #kf = KFold(n_splits=10)
    #print (kf.get_n_splits(X))
    #for training_index, test_index in kf.split(X):
    #    print("TRAIN:", training_index, "TEST:", test_index)
    #    X_training, X_test = X[training_index], X[test_index]
    #    Y_training, Y_test = Y[training_index], Y[test_index]
    
    
    #clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
    #          solver='svd', store_covariance=False, tol=0.0001)
    lda.fit(X_training, Y_training)
    
    predictions = lda.predict(X_test)
    
    print predictions
    print accuracy_score(Y_test, predictions)
=======
def prepare_data():
    """Prepare data for classifier to use"""
    #data, label = load_ta_data(), load_ta_target()
    data, label = load_own_data(), load_own_target()
    tra_x, tst_x = split_samples(data)
    tra_y, tst_y = split_samples(label)
    return (tra_x, tst_x, tra_y, tst_y)

def train_predict_and_results(data, clf):
    """Perform training, calculate predictions and show results"""
    tra_x, tst_x, tra_y, tst_y = data
    clf.fit(tra_x, tra_y)
    prd_y = clf.predict(tst_x)
    cnf = confusion_matrix(tst_y, prd_y)
    print ("Classifier: %s \tAccuracy score:%7.2f %%"
           "\tTN:%5d FP:%5d FN:%5d TP:%5d"
           % (clf.name, accuracy_score(tst_y, prd_y) * 100,
              cnf[0][0], cnf[0][1], cnf[1][0], cnf[1][1]))

def main():
    """The main method"""
    data = prepare_data()
    linear_discriminant_analysis(data)
    nearest_neighbors_classifier(data)
    support_vector_machine(data)
    gaussian_naive_bayes(data)
    logistic_regression(data)
    random_forest(data)
>>>>>>> 05e11c3b88b3fb5313f29e74125ab6fdd8fffd84

if __name__ == "__main__":
    main()
