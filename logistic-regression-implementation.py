
# coding: utf-8
from  model import lr as model
import pandas as pd

PATH_TRAINING = ''
PATH_TEST = ''

df_training = pd.read_csv(PATH_TRAINING)
df_test = pd.read_csv(PATH_TEST)



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

