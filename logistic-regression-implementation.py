
# coding: utf-8




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

