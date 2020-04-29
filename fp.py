import numpy, os.path
np = numpy
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, normalize
import scipy.stats
from timeit import default_timer as timer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from tensorflow import nn
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler


def load_abalone(path='abalone'):
    data = numpy.genfromtxt(os.path.join(path, 'abalone.data'), delimiter=',', dtype=str)
    X = data[:, :-1]
    Y = data[:,-1].reshape(-1,1)
    # One hot encoding for column sex
    sex_col = X[:, 0]
    ohe = OneHotEncoder(sparse=False)
    encoded_X = ohe.fit_transform(sex_col.reshape(-1, 1))
    # Concatenate encoded columns to original features
    X = np.delete(X, 0, 1)
    X = np.column_stack((encoded_X, X))

    return X.astype(float), Y.astype(float)


def grid_search_cv(estimator, param_grid, score, train_X, train_Y, test_X, test_Y, nn=False):
    # Grid Search Cross Validation for hyperparameter tuning
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    start = timer()

    if score is None:
        print("# Tuning hyper-parameters with default scoring: ")
    else:
        print("# Tuning hyper-parameters for %s" % score)
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=score, n_jobs=-1)
    warnings.filterwarnings("ignore", category=UserWarning)
    gscv.fit(train_X, train_Y)

    print("Grid scores on development set: <mean> <margin of error> for <parameter>")
    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Best parameters set found on development set:", gscv.best_params_)

    if not nn:
        print("\nDetailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_Y, gscv.predict(test_X)
        print(classification_report(y_true, y_pred, zero_division=0))
    end = timer()
    print("Time elapsed in Grid Search CV for hyperparameter tuning: %.2f seconds" % (end - start))
    return gscv.best_estimator_


def evaluate(model, test_features, test_labels):
    # Evaluate model performances (mean absolute percentage error)
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    predictions = model.predict(test_features)
    score = f1_score(test_labels, predictions, average='weighted')
    print('F1 Score = {:0.4f}.'.format(score))
    return score


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print("Mean: %0.3f (+/-%0.03f)" % (m, h))
    print("Confidence interval (95 percent): (%0.03f, %0.03f)" % (m + h, m - h))


def base_best_eval(base_model, best_model, test_X, test_Y):
    # Evaluate model performances before and after tuning hyperparameters
    # (compare base model with best GridSearchCV model)
    print("\nBase model performance: ")
    base_accuracy = evaluate(base_model, test_X, test_Y)
    print("Tuned model performance: ")
    gscv_accuracy = evaluate(best_model, test_X, test_Y)
    print('Improvement of {:0.2f}% after tuning hyperparameters.'.format(
        100 * (gscv_accuracy - base_accuracy) / base_accuracy))


def random_forest(train_X, test_X, train_Y, test_Y):
    # Get Random Forest with Bagging model
    rf = RandomForestClassifier(random_state=42)
    # Fit model on training data
    rf.fit(train_X, train_Y)

    # Tune hyperparameters with GridSearchCV
    param = {'n_estimators': [100, 115, 130, 145], 'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4], 'max_depth': [25, 30, 35]}
    score = 'f1_weighted'
    best_rf = grid_search_cv(rf, param, score, train_X, train_Y, test_X, test_Y)

    # Evaluate model performances before and after tuning hyperparameters
    base_best_eval(rf, best_rf, test_X, test_Y)

    # k-fold Cross Validation for Random Forest with Bagging after hyperparameter tuning
    cv_scores = []
    for i in range(10):
        warnings.filterwarnings("ignore", category=UserWarning)
        s = cross_val_score(best_rf, test_X, test_Y, cv=5, scoring='f1_weighted')
        cv_scores = np.append(cv_scores, s)

    # Performance of Random Forest with Bagging
    print("\nPerformance of Random Forest with Bagging: ")
    mean_confidence_interval(cv_scores)


def support_vector_machines(train_X, test_X, train_Y, test_Y):
    base_svm = SVC(kernel='rbf', cache_size=1900)
    linear_svm = SVC(kernel='linear', cache_size=1900)

    # Normalize data
    train_X = normalize(train_X)
    test_X = normalize(test_X)

    # Fit model on training data
    linear_svm.fit(train_X, train_Y)
    base_svm.fit(train_X, train_Y)

    # Tune hyperparameters with GridSearchCV
    param = {'C': np.logspace(-2, 3, 6), 'kernel': ['rbf', 'poly', 'sigmoid'],
             'gamma': np.logspace(-2, 1, 6)}
    score = 'f1_weighted'
    best_svm = grid_search_cv(base_svm, param, score, train_X, train_Y, test_X, test_Y)

    # Evaluate model performances before and after tuning hyperparameters
    base_best_eval(base_svm, best_svm, test_X, test_Y)

    # k-fold Cross Validation for Random Forest with Bagging after hyperparameter tuning
    cv_scores = []
    for i in range(10):
        warnings.filterwarnings("ignore", category=UserWarning)
        s = cross_val_score(best_svm, test_X, test_Y, cv=5, scoring='f1_weighted')
        cv_scores = np.append(cv_scores, s)

    # Performance of Random Forest with Bagging
    print("\nPerformance of Support Vector Machine: ")
    mean_confidence_interval(cv_scores)

    print("Base SVM Train Score: ", base_svm.score(test_X, test_Y))
    print("Best SVM Train Score: ", best_svm.score(test_X, test_Y))


def create_nn_model(epochs, learning_rate=0.01, momentum=0.0, init_mode='uniform'):
    # Define model
    model = Sequential()
    model.add(Dense(64, kernel_initializer=init_mode, activation=nn.relu, input_dim=10))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer=init_mode, activation=nn.relu))
    model.add(Dense(28, kernel_initializer=init_mode, activation=nn.softmax))
    # Define optimizer
    decay_rate = learning_rate / epochs
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def neural_network(train_X, test_X, train_Y, test_Y):
    # https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
    # Normalize data
    train_X = normalize(train_X)
    test_X = normalize(test_X)
    input_dim = train_X.shape[1]

    # Convert class vectors to binary class matrices
    num_classes = 28
    train_Y = to_categorical(train_Y, num_classes)
    test_Y = to_categorical(test_Y, num_classes)

    epochs = 30
    batch_size = input_dim

    # Create model
    nn = KerasClassifier(build_fn=create_nn_model, epochs=epochs, batch_size=batch_size, verbose=0)
    nn.fit(train_X, train_Y, verbose=0)

    # Tune hyperparameters with GridSearchCV
    param = {'epochs': [50, 60, 70], 'batch_size': [2**i for i in range(5, 8, 1)],
             'learning_rate': [0.01, 0.1, 1], 'momentum': [0.8, 0.9, 1.0]}
    score = None
    best_nn = grid_search_cv(nn, param, score, train_X, train_Y, test_X, test_Y, nn=True)
    best_nn.fit(train_X, train_Y, verbose=0)

    # Classification report (precision, recall, f1-score)
    print("\nDetailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    best_y_pred = best_nn.predict(test_X, verbose=0)
    y_true = np.argmax(test_Y, axis=1)
    print(classification_report(y_true, best_y_pred, zero_division=0))

    # Evaluate model performances before and after tuning hyperparameters
    # (compare base model with best GridSearchCV model)
    base_y_pred = nn.predict(test_X, verbose=0)
    print("\nBase model performance: ")
    base_accuracy = f1_score(y_true, base_y_pred, average='weighted')
    print('F1 Score = {:0.4f}.'.format(base_accuracy))
    print("Tuned model performance: ")
    gscv_accuracy = f1_score(y_true, best_y_pred, average='weighted')
    print('F1 Score = {:0.4f}.'.format(gscv_accuracy))
    print('Improvement of {:0.2f}% after tuning hyperparameters.'.format(
        100 * (gscv_accuracy - base_accuracy) / base_accuracy))

    # k-fold Cross Validation for Random Forest with Bagging after hyperparameter tuning
    cv_scores = []
    for i in range(10):
        warnings.filterwarnings("ignore", category=UserWarning)
        s = cross_val_score(estimator=best_nn, X=test_X, y=y_true, cv=5, scoring='f1_weighted')
        cv_scores = np.append(cv_scores, s)

    # Performance of Neural Network
    print("\nPerformance of Neural Network: ")
    mean_confidence_interval(cv_scores)


def main():
    np.random.seed(42)
    X, y = load_abalone()
    for ele in np.nditer(y, op_flags=['readwrite']):
        if ele == 29:
            ele[...] = ele - 2
        else:
            ele[...] = ele - 1

    # summarize class distribution
    y_1d = np.reshape(y, (y.shape[0],))
    print('Original dataset shape %s' % Counter(y_1d))

    # define pipeline
    oversample = RandomOverSampler(random_state=42)
    # transform the dataset
    X, y = oversample.fit_resample(X, y)

    num_data = 1000
    data = np.column_stack((X, y))
    print(data.shape)
    np.random.shuffle(data)
    data = data[:num_data, :]
    print(data.shape)
    X = data[:, :-1]
    y = data[:, -1]

    y_1d = np.reshape(y, (y.shape[0],))

    # summarize the new class distribution
    print('Resampled dataset shape %s' % Counter(y_1d))

    # Split dataset into training and testing sets
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest with Bagging
    rf_start = timer()
    random_forest(train_X, test_X, train_Y, test_Y)
    rf_end = timer()
    print("Time elapsed in Random Forest with Bagging: %.2f seconds" % (rf_end - rf_start))

    # Support Vector Machines
    svm_start = timer()
    support_vector_machines(train_X, test_X, train_Y, test_Y)
    svm_end = timer()
    print("Time elapsed in Support Vector Machines: %.2f seconds" % (svm_end - svm_start))

    # Neural Network
    nn_start = timer()
    neural_network(train_X, test_X, train_Y, test_Y)
    nn_end = timer()
    print("Time elapsed in Neural Network: %.2f seconds" % (nn_end - nn_start))


if __name__ == '__main__':
    main()
