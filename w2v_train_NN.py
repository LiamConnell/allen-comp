def main():
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
    import pybrain

    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.datasets import SupervisedDataSet
    from pybrain.supervised.trainers import BackpropTrainer

    from pybrain.tools.customxml.networkwriter import NetworkWriter
    from pybrain.tools.customxml.networkreader import NetworkReader

    def remove_punctuation(s):
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        s = s.lower()
        return s
    #TODO: fix this or dont use
    def get_longword(s, n):
        s = re.split(' ', s)
        s =  [w for w in s if not w in stop]
        return heapq.nlargest(n, s, key=len)

    def get_uniqwds(row, ans):
        answerwords  = []
        for col in row['answerA':'answerD']:
            col = re.split(' ', col)
            for c in col:
                answerwords.append(c)
        if ans == 'a':
            col = row['answerA']
        if ans == 'b':
            col = row['answerB']
        if ans == 'c':
            col = row['answerC']
        if ans == 'd':
            col = row['answerD']
        col = re.split(' ', col)
        uniq = [word for word in col if answerwords.count(word) == 1]
        return ' '.join(uniq)

    def get_rarewrd(s):
        s = re.split(' ', s)
        s =  [w for w in s if not w in stop]
        freqs = [words.count(x) for x in s]
        dd= [i for i,x in enumerate(freqs) if x == min(freqs)]
        try:
            return [s[dd[0]]]
        except:
            return []

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

    def convert_back2arrays(s):
        s = np.fromstring(s[3:-1], sep = ' ')
        return s

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
                    wd = words[x-500:x+500]
                    ct = wd.count(answer[0])
                    waka = (waka + ct)#/(words.count(answer[0])/2)    #trying to skew for 
                kkk.append(waka)
            mx = max(kkk)
            print(kkk)
            best = [i for i, j in enumerate(kkk) if j == mx]
            print(row['correctAnswer'])
            print(best)
            if len(best) > 1:
                best = best[0]

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

    #########################
    #########################


    data  = pd.read_csv('../input/training_set_(add_cols)n001.csv')

    data.qvec = data.qvec.apply(convert_back2arrays)
    data.tvec = data.tvec.apply(convert_back2arrays)

    isub = [len(data.tvec[i]) == 300 for i in data.index]
    data = data.iloc[isub]
    jsub = [len(data.qvec[i]) == 300 for i in data.index]
    data = data.iloc[jsub]

    qarray = np.array([data.qvec[i] for i in data.index])
    tarray = np.array([data.tvec[i] for i in data.index])

    net = buildNetwork(300, 3, 300)#, bias=True, hiddenclass=TanhLayer)

    ds = SupervisedDataSet( 300, 300 )
    ds.setField( 'input', qarray )
    ds.setField( 'target', tarray )

    trainer = BackpropTrainer( net, ds )

    trainer.trainUntilConvergence( verbose = True, validationProportion = 0.05, maxEpochs = 500, continueEpochs = 10 )

    NetworkWriter.writeToFile(net, '../input/NN(.05-500-10)_001.xml')



if __name__ == '__main__':
    main()