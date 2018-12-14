
# coding: utf-8

# In[21]:


import numpy as np


# In[6]:


class model(object):
    
    """ Classificador Linear - Regressão Logística

    Parâmetros:
    ------------
    eta : float
        Taxa de aprendizagem (entre 0.0 e 1.0)
    n_iter : int
        N. de interações na fase de treinamento

    Atributos
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
        """ Activate the logistic neuron"""
        z = self.net_input(X)
        return self._sigmoid(z)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]            
        
        Returns
        ----------
          Class 1 probability : float
        
        """
        return activation(X)

    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]            
        
        Returns
        ----------
        class : int
            Predicted class label.
        
        """
        # equivalent to np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# In[16]:


get_ipython().system("ls '/home/rafael/USP/drive/Data/Dataframe/b5post'")


# In[19]:


import pandas as pd

df_training = pd.read_csv('/home/rafael/USP/drive/Data/Dataframe/b5post/gender_pt_training_df.csv')
df_test = pd.read_csv('/home/rafael/USP/drive/Data/Dataframe/b5post/gender_pt_test_df.csv')


# In[87]:


bow = []
max_features = 10
word_dict = dict()

#max_features = np.array([df_training['text'].values[i].split(" ") for i in range(len(df_training))]).mean()
for i in range(len(df_training)):
    values = df_training['text'].values[i].split(" ")
    limited_values = []
    for j in range(0, max_features):
        try:
            v = values[j]
            v = v.replace("\n", "")
            v = v.lower()
        except:
            v = 0
            
        if v not in word_dict:
            word_dict[v] = 1
        else:
            word_dict[v] = word_dict[v] + 1
        
        limited_values.append(v)
        
    bow.append(limited_values)


# In[88]:


np.array(bow).shape, max_features, len(word_dict), len(bow)


# In[207]:


tf_vect = []
for b in bow:
    tf_dict = {}
    bowCount = len(b)
    for word,val in word_dict.items():    
        tf_dict[word] = float(bowCount)/word_dict[word]
    tf_vect.append(tf_dict)


# In[208]:


import math

idf_dict = {}
N = len(bow)

idf_dict = dict.fromkeys(word_dict.keys(), 0)

for word, val in word_dict.items():          
    idf_dict[word] += val

for word, val in idf_dict.items():    
    idf_dict[word] = math.log10(N / float(val))


# In[209]:


tfidf_vect = []
for tf in tf_vect:
    tfidf = []
    
    for word, val in tf.items():       
        
        tfidf.append(val*idf_dict[word])        
    
    tfidf_vect.append(tfidf)    


# In[210]:


import pandas as pd
X = pd.DataFrame(tfidf_vect)
X = X.fillna(1)


# In[225]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:].min() - 1, X[:].max() + 1
    x2_min, x2_max = X[:].min() - 1, X[:].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# In[226]:


def labels(x):
    if x == 'female':
        return 0
    else: 
        return 1
y = df_training['gender'].apply( labels )
y = y.values


# In[228]:


X.shape, y.shape


# In[229]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

lr = model(n_iter=500, eta=0.2).fit(X, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()


# In[230]:


plot_decision_regions(X, y, classifier=lr)
plt.title('Logistic Regression - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
