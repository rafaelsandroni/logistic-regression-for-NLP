# coding: utf-8

import numpy as np

class lr(object):
    
    """ Classificador Linear - Regressão Logística

    Parameters:
    ------------
    eta : float
        Taxa de aprendizagem (entre 0.0 e 1.0)
    n_iter : int
        N. de interações na fase de treinamento

    Attributes
    -----------
    w_ : 1d-array
        Pesos após a fase de treinamento
    cost_ : list
        Custo em cada época

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """ Training

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features] => features
        y : array-like, shape = [n_samples] => targets
        
        n_samples = number of samples
        n_features = number of features
            

        Return
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1]) # features
        self.cost_ = []       
        for i in range(self.n_iter): # epochs
            y_val = self.activation(X) 
            errors = (y - y_val) 
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.eta * neg_grad
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):        
        z = self.net_input(X)
        return self._sigmoid(z)
    
    def predict_proba(self, X):        
        return activation(X)

    def predict(self, X):        
        # np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(self.net_input(X) >= 0.0, 1, 0)


