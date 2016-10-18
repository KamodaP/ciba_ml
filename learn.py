import numpy as np
import matplotlib.pyplot as plt
import dateutil
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from settings import params

_PATH = os.path.join('.', 'src')

def regression(train_f = 'regression_train.csv', test_f = 'regression_test.csv'):
    with open(os.path.join(_PATH, train_f)) as train:
        train_t = train.read()
    with open(os.path.join(_PATH, test_f)) as test:
        test_t = test.read()

    train_set_X = []
    train_set_Y = []
    test_set_X = []
    test_set_Y = []
    for line in train_t.split('\n')[1:]:
        train_set_X.append(line.split(',')[0])
        train_set_Y.append(line.split(',')[1])
    for line in test_t.split('\n')[1:]:
        test_set_X.append(line.split(',')[0])

    tr_len = len(train_set_X)
    ts_len = len (test_set_X)

    train_a_X = np.array(train_set_X, np.float64).reshape(tr_len, 1)
    train_a_Y = np.array(train_set_Y, np.float64)
    test_a_X = np.array(test_set_X, np.float64).reshape(ts_len, 1)

    clf = MLPRegressor(**params.reg_params)
    svr = clf.fit(train_a_X, train_a_Y)

    pred = svr.predict(test_a_X)

    lw = 2
    plt.figure()
    plt.scatter(train_a_X, train_a_Y, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(test_a_X, pred, color='navy', lw=lw, label='Linear model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend()

def classification(train_f = 'classification_train.csv', test_f = 'classification_test.csv'):
    with open(os.path.join(_PATH, train_f)) as train:
        train_t = train.read()
    with open(os.path.join(_PATH, test_f)) as test:
        test_t = test.read()

    train_set_feat = []
    train_set_cls = []
    test_set_feat = []
    for line in train_t.split('\n')[1:-2]:
        train_set_feat.append((line.split(',')[0], line.split(',')[1]))
        train_set_cls.append(line.split(',')[2])
    for line in test_t.split('\n')[1:-2]:
        test_set_feat.append((line.split(',')[0], line.split(',')[1]))

    tr_len = len(train_set_feat)
    ts_len = len (test_set_feat)

    train_a_feat = np.array(train_set_feat, np.float64).reshape(tr_len, 2)
    train_a_cls = np.array(train_set_cls, np.float64)
    test_a_feat = np.array(test_set_feat, np.float64).reshape(ts_len, 2)

    clf = MLPClassifier(**params.class_params)
    svr = clf.fit(train_a_feat, train_a_cls)

    pred = svr.predict(test_a_feat)
    h = 0.2

    x_min, x_max = train_a_feat[:, 0].min() - 1, train_a_feat[:, 0].max() + 1
    y_min, y_max = train_a_feat[:, 1].min() - 1, train_a_feat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    Z = svr.predict(np.c_[xx.ravel(), yy.ravel()]) 

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(train_a_feat[:, 0], train_a_feat[:, 1], c=train_a_cls, cmap=plt.cm.Paired)

if __name__ == "__main__":
    regression()
    classification()
    plt.show()
