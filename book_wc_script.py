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
from nltk.corpus import stopwords 




def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s
#TODO: fix this or dont use
def get_longword(s, n):
    s = re.split(' ', s)
    s =  [w for w in s if not w in stops]
    return heapq.nlargest(n, s, key=len)

def get_rarewrd(s):
    s = re.split(' ', s)
    s =  [w for w in s if not w in stops]
    freqs = [words.count(x) for x in s]
    dd= [i for i,x in enumerate(freqs) if x == min(freqs)]
    return [s[dd[0]]]

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
    
def run_through_theta(v):
    return v.dot(theta.as_matrix())

def get_winner_from_avecs(row):
	try:	
		dists = []
		for col in row['answerA':'answerD']:
		#TODO: if 'not' in words: 1-avec
			avec = get_avg_vec(col)
			#if 'not' in re.split(' ',col):
			#	avec = 1-avec
			#	print('NOT')
			#why nan errors?
			if sum(np.isfinite(avec)) < 300:
				print('avec')
				print(avec)
			if  sum(np.isfinite(row['pvec'])) < 300:
				print('pvec')
				print(row['pvec'])
			dist = cosine(avec, row['pvec'])
			dists.append(dist)
		m = np.nanmin(dists)
		best = [i for i, j in enumerate(dists) if j == m]
		if best == [0]:
			return 'A'
		if best == [1]:

			return 'B'
		if best == [2]:
			return 'C'
		if best == [3]:
			return 'D'
		else:
			print('returning C becuase no best cosine')
			print(len(best))
			print(dists)
			print(m)
			return 'C'
	except:
		print('some error, returning C')
		return 'C'
    
    
    
    
def word_clusters(row):
    if str(row.id)[-2:] == '01':
        print(row.id)
        print('\n\n\n\n')
    #print(row.correctAnswer)
    try:
        keywds = row.keyword

        word_pos = {}
        for keyw in keywds:
            ls = [i for i,x in enumerate(words) if x == keyw]
            #print(ls)
            word_pos[keyw] = ls

        spots = []
        for key1 in keywds:
            print(key1)
            matches = {}
            for z in word_pos[key1]:
                dists=[]
                for key_ in keywds:
                    d= abs(min(word_pos[key_], key=lambda x:abs(x-z))-z)
                    dists.append(d)
                #print(z)
                #print(dists)
                dists.remove(max(dists))
                matches[z] = sum(dists)

            zzz= [k for k, v in matches.items() if v ==min(matches.values())]
            spots.append(zzz[0])
        print(spots)
        answers = row['akeyword':'dkeyword']
        kkk = []
        for answer in answers:
            print(answer[0])
            waka = 0
            for x in spots:
                wd = words[x-1000:x+1000]
                ct = wd.count(answer[0])
                waka = (waka + ct)#/(words.count(answer[0])/2)    #trying to skew for 
            kkk.append(waka)
        mx = max(kkk)
        print(kkk)
        best = [i for i, j in enumerate(kkk) if j == mx]
        if best == [0]:
            return 'A'
        if best == [1]:
            return 'B'
        if best == [2]:
            return 'C'
        if best == [3]:
            return 'D'
        else:
            print('returning C becuase no max')
            return 'C'
    except:
        print('returning C becuase of BIG PROBLEM')
        return 'C'
    
    ##############################
    ##############################
    ##############################
    
#data  = pd.read_csv('../input/training_set.tsv', '\t')
data  = pd.read_csv('../input/validation_set.tsv', '\t')

stop = stopwords.words('english')


file = open('../input/Concepts.txt', 'r')
words = list(file.read().split())
words = [word.strip(string.punctuation) for word in words]
words = [w for w in words if not w in stop]

data.question = data.question.apply(remove_punctuation)
data.answerA = data.answerA.apply(remove_punctuation)
data.answerB = data.answerB.apply(remove_punctuation)
data.answerC = data.answerC.apply(remove_punctuation)
data.answerD = data.answerD.apply(remove_punctuation)

data['keyword'] =data['question'].apply(get_longword, args = (4,))

data['akeyword'] =data['answerA'].apply(get_rarewrd)
data['bkeyword'] =data['answerB'].apply(get_rarewrd)
data['ckeyword'] =data['answerC'].apply(get_rarewrd)
data['dkeyword'] =data['answerD'].apply(get_rarewrd)

data['guess'] = data.apply(word_clusters, axis = 1)

sample = pd.read_csv('../input/sample_submission.csv')
sample['correctAnswer'] = data['guess']
sample.to_csv('../output/word_cluster_001.csv', index=False)
