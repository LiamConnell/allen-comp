import time
import pandas as pd
from wikiapi import WikiApi
wiki = WikiApi()
import re
#import word2vec

def get_longword(s):
    return max(re.split(' ', s), key=len)


def get_wiki(k):
    try:
        return wiki.get_article(wiki.find(k)[0]).summary
    except:
        return []


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



start = time.time()
#data  = pd.read_csv('../input/training_set.tsv', '\t')
data  = pd.read_csv('../input/validation_set.tsv', '\t')
lap1 = time.time()
print('data gathered: %s' % (lap1 - start))


data['keyword'] =data['question'].apply(get_longword)
lap2 = time.time()
print('longword: %d' % (lap2 - lap1))

data['words'] = data.keyword.apply(get_wiki)
lap3 = time.time()
print('get wiki: %d' % (lap3 - lap2))

data['comp'] = data.apply(compete, axis = 1)
lap4 = time.time()
print('comp: %d' % (lap4 - lap3))
data['guess'] = data.comp.apply(answerit)
lap5 = time.time()
print('guess: %d' % (lap5 - lap4))
data['sub'] = data.guess.apply(convert)
lap6 = time.time()
print('get wiki: %d' % (lap6 - lap5))


sample = pd.read_csv('../input/sample_submission.csv')
sample['correctAnswer'] = data['sub']
lap7 = time.time()
print('read and sub: %d' % (lap7 - lap6))
sub  = open('../output/submission.csv', 'w')
sample.to_csv('../output/submission.csv', index=False)
lap8 = time.time()
print('write csv: %d' % (lap8 - lap7))


