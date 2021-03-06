import time
import pandas as pd
from wikiapi import WikiApi
wiki = WikiApi()
import re
from gensim.models import Word2Vec
#import word2vec
from scipy.spatial.distance import cosine
import numpy as np
import heapq
import string



def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

#TODO: fix this or dont use
def get_longword(s):
    s = re.split(' ', s)
    return heapq.nlargest(5, s, key=len)

#TODO: make this work with one word/period
def get_avg_vec(words):
	try:
		words = re.split(' ', words)
	except:
		pass
	nwords = 0
	featureVec = np.zeros((num_features,))#,dtype="float32")
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])
	if nwords > 0:
		featureVec = np.divide(featureVec,nwords)
	return featureVec

def get_tvec(row):
    if row['correctAnswer'] == 'A':
        a = row['aavec']
        return a
    elif row['correctAnswer'] == 'B':
        a = row['bavec']
        return a
    elif row['correctAnswer'] == 'C':
        a = row['cavec']
        return a
    elif row['correctAnswer'] == 'D':
        a = row['davec']
        return a
    
def cost_function(data, theta):
    dists =  [cosine(data.iloc[i]['qvec'].dot(theta), data.iloc[i]['tvec']) for i in data.index]
    return np.nansum(dists)/len(data)

def update_theta(data, theta, alpha):
    x = np.array([data.iloc[i]['qvec'] for i in data.index])    
    xTrans = x.transpose()
    ttheta = theta.copy()
    for m in range(300):
        #print(m)
        y = np.array([data.iloc[i]['tvec'][0] for i in data.index])
        hypothesis = np.dot(x, theta[m,])
        loss = hypothesis - y
        gradient = np.dot(xTrans, loss) / len(data)  
        ttheta[m,] = ttheta[m,] - alpha * gradient  
        
    return ttheta

def gradient_descent(data,theta, alpha = .01, iter = 4):
    #theta = np.identity(300)
    cost_hist = []
    cost = cost_function(data, theta)
    print(cost)
    cost_hist.append(cost)
    for i in range(iter):
        alpha_1 = alpha#/(i+1)
        theta = update_theta(data, theta, alpha_1)
        cost = cost_function(data, theta)
        print(cost)
        cost_hist.append(cost)
    print(cost_hist)
    return theta, cost_hist
    
    
    
#####function:    
start = time.time()
model = Word2Vec.load_word2vec_format('/Users/liamconnell/Downloads/GoogleNews-vectors-negative300.bin', binary = True)
lap1 = time.time()
print('data gathered: %s' % (lap1 - start))


index2word_set = set(model.index2word)
num_features = 300

#TODO: mod training_set
data  = pd.read_csv('../input/training_set.tsv', '\t')

###REMOVE PUNCTUATION###
data.question = data.question.apply(remove_punctuation)
data.answerA = data.answerA.apply(remove_punctuation)
data.answerB = data.answerB.apply(remove_punctuation)
data.answerC = data.answerC.apply(remove_punctuation)
data.answerD = data.answerD.apply(remove_punctuation)

#only if I do keyword
data['keyword'] =data['question'].apply(get_longword)

#I get to choose this
data['qvec'] = data.keyword.apply(get_avg_vec) 

data['aavec'] = data['answerA'].apply(get_avg_vec)
data['bavec'] = data['answerB'].apply(get_avg_vec)
data['cavec'] = data['answerC'].apply(get_avg_vec)
data['davec'] = data['answerD'].apply(get_avg_vec)

tvec = []
for i in range(len(data)):
    tvec.append(get_tvec(data.iloc[i,:]))
data['tvec'] = tvec

#theta = pd.read_csv('../input/qkeywd_theta_50-a_e-10.csv').as_matrix()
theta = np.identity(300) #* -1
theta, cost_hist = gradient_descent(data,theta,alpha = .01, iter = 1000)
theta = pd.DataFrame(theta)
theta.to_csv('../input/qkeywd-5_theta_1000-a_e-100.csv', index = False)