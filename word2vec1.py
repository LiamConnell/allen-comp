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

start = time.time()
model = Word2Vec.load_word2vec_format('/Users/liamconnell/Downloads/GoogleNews-vectors-negative300.bin', binary = True)
lap1 = time.time()
print('data gathered: %s' % (lap1 - start))



#not used
def get_longword(s):
    return heapq.nlargest(2, s, key=len)
#max(re.split(' ', s), key=len)

#not used
#def get_wiki(k):
 #   try:
  #      return wiki.get_article(wiki.find(k)[0]).content
   # except:
    #    return []

def get_wiki(q):
	try:
		return wiki.get_article(wiki.find(get_longword(q)[0])[0]).content
	except:
		try:
			return wiki.get_article(wiki.find(get_longword(q)[1])[0]).content
		except:
			print('neither works')
			return []



index2word_set = set(model.index2word)
num_features = 300



def get_avg_vec(words):
	words = re.split(' ', words)
	nwords = 0
	featureVec = np.zeros((num_features,),dtype="float32")
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])
	featureVec = np.divide(featureVec,nwords)
	return featureVec

def get_winner_from_avecs(row):
	try:	
		dists = []
		for col in row['answerA':'answerD']:
		#TODO: if 'not' in words: 1-avec
			avec = get_avg_vec(col)
			if 'not' in re.split(' ',col):
				avec = 1-avec
				print('NOT')
			#why nan errors?
			if sum(np.isfinite(avec)) < 300:
				print('avec')
				print(avec)
			if  sum(np.isfinite(row['qvec'])) < 300:
				print('qvec')
				print(row['qvec'])
			dist = cosine(avec, row['qvec'])
			dists.append(dist)
		m = min(dists)
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
			return 'C'
	except:
		#print('some error, returning C')
		return 'C'
		




def overlap(answw, words):
    count = 0
    for word in re.split(' ', answw):
        if word in words:
            count = count+1
    return count

def compete(row):
    lis = []
    #print(row)
    for col in row['answerA':'answerD']:
        lis.append(overlap(col, row['words']))
    return lis


def answerit(lis):
    #print(lis)
    m = max(lis)
    return [i for i, j in enumerate(lis) if j == m]


def convert(g):
    if len(g) == 1:
        if g == [0]:
            return 'A'
        if g == [1]:
            return 'B'
        if g == [2]:
            return 'C'
        if g == [3]:
            return 'D'
    else:
        return 'C'


ccount = 0
start = time.time()
#data  = pd.read_csv('../input/training_set.tsv', '\t')
#data  = pd.read_csv('../input/validation_set.tsv', '\t')
data = pd.read_csv('../input/validation_set_mod.csv')
lap1 = time.time()
print('data gathered: %s' % (lap1 - start))


#data['keyword'] =data['question'].apply(get_longword)
lap2 = time.time()
#print('longword: %d' % (lap2 - lap1))

data['words'] = data['question'].apply(get_wiki)
lap3 = time.time()
print('get wiki: %d' % (lap3 - lap2))

#save dataset with wiki
data.to_csv('../input/validation_set_mod2.csv')

data['qvec'] = data.words.apply(get_avg_vec)
lapa = time.time()
ccount = 0
print('get qvec: %d' % (lapa - lap3))
print(ccount)
data['closest_avec'] = data.apply(get_winner_from_avecs, axis=1)

#data['comp'] = data.apply(compete, axis = 1)
#lap4 = time.time()
#print('comp: %d' % (lap4 - lap3))
#data['guess'] = data.comp.apply(answerit)
#lap5 = time.time()
#print('guess: %d' % (lap5 - lap4))
#data['sub'] = data.guess.apply(convert)
#lap6 = time.time()
lapb = time.time()
print('get wiki: %d' % (lapb - lapa))


sample = pd.read_csv('../input/sample_submission.csv')
sample['correctAnswer'] = data['closest_avec']
lap7 = time.time()
print('read and sub: %d' % (lap7 - lapb))
#sub  = open('../output/submission.csv', 'w')
sample.to_csv('../output/w2v_avg.csv', index=False)
lap8 = time.time()
print('write csv: %d' % (lap8 - lap7))

print('ccount')
print(ccount)
