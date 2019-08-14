#!/usr/bin/env python
# coding: utf-8

# In[87]:


#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
import re
from collections import defaultdict

#--- CONSTANTS ----------------------------------------------------------------+


class word2vec():
    def __init__ (self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass
    
    
    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                
                #w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)


    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec


    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
                

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)  
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass


    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))     # context matrix
        
        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:

                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)
                
                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
                
            print('EPOCH:',i, 'LOSS:', self.loss)
        
        return self.word_index, self.index_word, self.w1, self.v_count
        pass


    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda sim :sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
            
        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):
        
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda x:x[1], reverse=True)
                
        for word, sim in words_sorted[:top_n]:
            print(word, sim)        
        pass
    # 保存: self.word_index, w1, v_count
    


# In[88]:


#モジュールのインポート
import pandas as pd
import datetime


# In[89]:


df = pd.read_csv("user.csv",chunksize=100)
df = df.get_chunk()

a = []
b = []

#Remove hour, minutes, seconds
for i in range(len(df)):
    a.append(datetime.datetime.strptime(df["date_time"][i], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    b.append(datetime.datetime.strptime(df["date_time"][i], '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S'))

a = pd.DataFrame(a)
b = pd.DataFrame(b)
a = a.rename(columns={0:"date"})
b = b.rename(columns={0:"time"})

df = pd.concat([df, a, b], axis=1, join="inner")

del a, b


# In[90]:


data_hira = df.sort_values(["date","id"])
dum = []

for i in range(len(data_hira)):
        dum.append(str(data_hira["id"][i]) + data_hira["date"][i])
dum = pd.DataFrame(dum)
dum = dum.rename(columns={0: 'new_name'})
dummy = pd.concat([data_hira, dum], axis=1, join='inner')
data_hira = dummy.sort_values(by=["time","new_name"])

del dummy, dum, df

keyword_ls = data_hira["keyword"].values
id_ls = data_hira["new_name"].values

del data_hira


# In[91]:


main_list = []
key = id_ls[0]
dum_ls = []

for i in range(len(keyword_ls)):
    if id_ls[i] == key:
        dum_ls.append(keyword_ls[i])
    else:
        key = id_ls[i]
        dum_ls.append("...")
        main_list.append(dum_ls)
        dum_ls = []
        dum_ls.append(keyword_ls[i])

del dum_ls, keyword_ls, id_ls, key


# In[92]:


#--- EXAMPLE RUN --------------------------------------------------------------+

settings = {}
settings['n'] = 5              # dimension of word embeddings
settings['window_size'] = 2         # context window +/- center word
settings['min_count'] = 0           # minimum word count
settings['epochs'] = 5000          #5000 number of training epochs
settings['neg_samp'] = 10          # number of negative words to use during training
settings['learning_rate'] = 0.01    # learning rate
np.random.seed(0)                   # set the seed for reproducibility

corpus = main_list

# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, corpus)

# train word2vec model
model = w2v.train(training_data)


#--- END ----------------------------------------------------------------------+


# In[94]:


#modelをcsvとして出力
keyword_list = []
sim_list = []
v_count_list = []
for i in range(len(model[1])):
    keyword_list.append(model[1][i])

sub_list=[]
for i, item in np.ndenumerate(model[2]):
    sub_list.append(item)
for i in range(len(model[2])):
    sim_list.append(sub_list)
    
for i in range(len(model[2])):
    v_count_list.append(model[3])

train_dict = {"keyword":keyword_list,"sim":sim_list,"v_count":v_count_list}
train_df = pd.DataFrame.from_dict(train_dict)

del keyword_list, sim_list, v_count_list, sub_list, train_dict
# CSV ファイル (employee.csv) として出力
train_df.to_csv("train.csv")


# # データ保存

# In[101]:


u"""
train_df = pd.read_csv("train.csv")


#ディクショナリに変換
# word_index, index_word 作成
word_index = {v:k for k, v in model_d1.items()}
index_word = train_df["keyword"].to_dict()

# w1 生成
w1 = []
for i in range(len(model[2])):
    w1.append(list(model[2][i]))
    
w1 = np.array(w1)

# n_count 生成
v_count = train_df.v_count[0]

def word_sim(a,index_word,b,c, word, top_n):

    w1_index = a[word]
    v_w1 = b[w1_index]


    # CYCLE THROUGH VOCAB
    word_sim = {}
    for i in range(c):
        v_w2 = b[i]
        theta_num = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_num / theta_den

        word = index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda x:x[1], reverse=True)

    for word, sim in words_sorted[:top_n]:
        print(word, sim)        
    pass
    
word_sim(word_index,index_word,w1,n_count,'お中元',10)
"""

