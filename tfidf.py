

def transform(X):

  bow = []
  max_features = 100
  word_dict = dict()

  for i in range(len(X)):
      values = X['text'].values[i].split(" ")
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
      
  return tfidf_vect
