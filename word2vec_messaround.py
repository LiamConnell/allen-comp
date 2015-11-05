# IPython log file

import time
import pandas as pd
from wikiapi import WikiApi
wiki = WikiApi()
import re
from gensim.models import Word2Vec
#import word2vec
start = time.time()
model = Word2Vec.load_word2vec_format('/Users/liamconnell/Downloads/GoogleNews-vectors-negative300.bin', binary = True)
lap1 = time.time()
print('data gathered: %s' % (lap1 - start))
data = pd.read_csv('../input/validation_set_mod.csv')
data.head(1)
data[0]
data[[0]]
data.index
data['0']
data[data.index[0]]
data.iloc[0]
row = data.iloc[0]
words = row['words']
words.head()
words
'is' in words
'sdfsdfsdfsdf' in words
import re
words_ = re.split(' ', words)
words_[:5]
model
model['short']
model['shorsdfsdft']
index2word_set = set(model.index2word)
get_ipython().magic('logstart word2vec_messaround.py')
for word in words:
    if word in index2word_set: 
        nwords = nwords + 1.
        featureVec = np.add(featureVec,model[word])
        
nwords = 0
for word in words:
    if word in index2word_set: 
        nwords = nwords + 1.
        featureVec = np.add(featureVec,model[word])
        
import numpy as np
for word in words:
    if word in index2word_set: 
        nwords = nwords + 1.
        featureVec = np.add(featureVec,model[word])
        
featureVec = np.zeros((num_features,),dtype="float32")
num_features = 300
featureVec = np.zeros((num_features,),dtype="float32")
for word in words:
    if word in index2word_set: 
        nwords = nwords + 1.
        featureVec = np.add(featureVec,model[word])
        
featureVec = np.divide(featureVec,nwords)
featureVec
model[featureVec]
model.similarity(featureVec, 'human')
quit()
