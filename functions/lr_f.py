import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss_regularized(w, X, y, penalty='l2', C=1.0):
    eps = 1e-8
    z = np.dot(X, w)
    p = 1 / (1 + np.exp(-z))

    
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    
    if penalty == 'l2':
        reg = 0.5 * np.sum(w ** 2)
    elif penalty == 'l1':
        reg = np.sum(np.abs(w))
    else:
        raise ValueError("penalty must be 'l1' or 'l2'")
    loss += (1 / C) * reg

    
    grad = np.dot(X.T, (p - y)) / len(y)

    
    if penalty == 'l2':
        grad += (1 / C) * w
    elif penalty == 'l1':
        grad += (1 / C) * np.sign(w)

    return loss, grad


def gradient_descent(X, y, loss_func, learning_rate=0.01, epochs=500):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        loss, gradient = loss_func(w, X, y)
        w -= learning_rate * gradient
    return w


class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', C=1.0, learning_rate=0.001, epochs=500, solver = 'liblinear'): # era 500, le puse 1000
        self.penalty = penalty
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.solver = solver

    def fit(self, X, y):
        self.w = gradient_descent(
            X, y,
            lambda w, X, y: cross_entropy_loss_regularized(
                w, X, y,
                penalty=self.penalty,
                C=self.C
            ),
            learning_rate=self.learning_rate,
            epochs=self.epochs
        )
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p = sigmoid(np.dot(X, self.w))
        return np.array([1 - p, p]).T

    def predict(self, X):
        return (sigmoid(np.dot(X, self.w)) >= 0.5).astype(int)