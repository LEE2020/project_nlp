#!/bin/env/python
# -*- encoding:utf-8 -*-
#/**********************************************************
# * Author        : 07zhiping 
# * Email         : 07zhiping@gmail.com 
# * Last modified : 2020-10-31 13:35
# * Filename      : tensorflow_sentences_embedding_supervised.py
# * Description   : 1) data source : https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb
#                   2) input data ： 外卖正向评论/ 负向评论 共10000+条（7000 + 4000 ）
#                   3) label ： 正向/负向
#                   4) output  ： label
#                   5) model_layer.weights -> vocal embedding 
# ps. 中文停用词 https://github.com/goto456/stopwords/blob/master/cn_stopwords.txt 
# * *******************************************************/

import tensorflow as tf
from  tensorflow.keras.preprocessing.text import Tokenizer # tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding (left or right)
import jieba 
import os 
import random 
import numpy as np 

class sentiment_classification:
        def __init__(self):
            self.sentences = []
            self.label = []
            self.stop_words = []
            self.word_index = {}
            self.final_training = []
            self.final_train_label = []
            self.final_test = []
            self.final_test_label = [] 

            for line in open('../chinese_data/cn_stopwords.txt','r',encoding='utf=8'):
                lines = line.strip().split()[0]
                if lines: self.stop_words.append(lines)
                 

        def load_data(self, data_path):
            files = open(data_path,'r',encoding='utf-8')
            flag = True 
            for line in files:
                lines = line.strip().split(',',1)
                if flag : flag = False ; continue
                if len(lines) < 2: continue
                cut_words = jieba.cut(lines[1],cut_all=False) # 分词 
                filter_words = []
                for i in list(cut_words):
                    if i in self.stop_words:continue   # 去除中文停用词
                    filter_words.append(i)
                #print(filter_words) 
                self.sentences.append(filter_words)
                self.label.append(int(lines[0]))
        def balanced_samples(self):
            Nsamples = len(self.label)
            pos_samples = sum(self.label)
            neg_samples = Nsamples - pos_samples 
            N = 8000 
            tmp = list(zip(self.sentences,self.label))
            random.shuffle(tmp)  # 消除序列因素对模型的影响
            pos_cnt,neg_cnt = 0, 0
            
            
            for i,j  in tmp:
                if pos_cnt >=4000 and j == 1 or neg_cnt >= 4000 and j ==0 :
                    self.final_test.append(i)
                    self.final_test_label.append(j)    
                    continue
                 
                self.final_training.append(i)  
                self.final_train_label.append(j)
                if j == 1: pos_cnt +=1 
                else: neg_cnt += 1
                #if pos_cnt >=4000 and neg_cnt >=2000 : break 
                
            self.sentences = [] #释放无用的数据
            self.label = [] 
            #print(len(self.final_training),len(self.final_train_label),len(self.final_test),len(self.final_test_label),sum(self.final_train_label)) 
        def tokenizer_data(self):
            vocab_size = 10000  # seleceted by itself throuth word frequency 
            embedding_dim = 4   # each word embedding dimension
            max_length = 120    # each sentence contain 120 words at most 
            trunc_type='pre'  # trunc_type on post less than 120 
            pad_type = 'pre' 
            oov_tok = "<OOV>"  # words not contained in vocab_size words 
            # fit tokenizer on train data
            tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) 
            tokenizer.fit_on_texts(self.final_training)
            self.word_index = tokenizer.word_index
            # final training data/ final test data 
            final_training  = tokenizer.texts_to_sequences(self.final_training)
            pad_training  = pad_sequences(final_training,maxlen=max_length, truncating=trunc_type,padding='pre')
            final_test = tokenizer.texts_to_sequences(self.final_test)
            pad_test = pad_sequences(final_test,maxlen = max_length, truncating = trunc_type,padding = pad_type) 
            

            # type -> np.array()
            pad_training = np.array(pad_training)
            self.final_train_label = np.array(self.final_train_label)
            pad_test = np.array(pad_test)
            self.final_test_label = np.array(self.final_test_label) 
            return pad_training,pad_test 
          #  print(len(final_training),len(pad_training),len(pad_training[0]))
          #  print(len(final_test),len(pad_test),len(pad_test[0]))     
        def run(self):
            vocab_size = 10000 
            embedding_dim = 4
            max_length = 120  
            pad_training, pad_test = self.tokenizer_data()
            model = tf.keras.Sequential([
                    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(6,activation='relu'),
                    tf.keras.layers.Dense(1,activation='sigmoid') ])
            model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
            model.summary()

            num_epochs = 10 
            model.fit(pad_training,self.final_train_label,batch_size = 50 ,epochs = num_epochs,validation_data= (pad_test,self.final_test_label))
        
        def decode_string(self,text):
            reverse_word_index = [(index,word) for (word,index) in self.word_index.items()]
            return ' '.join([reverse_word_index.get(i, '?') for i in text]) 
        def get_weight(self):
            # embedding layer ===> layer[0]
            embedding_Matrix = {}
            weights = model.layer[0].get_weights()
             
            reverse_word_inde = dict([(index,word) for (word,index) in self.word_index.items()])
            for num in range(1,vocab_size):  # word_index from 1: vocab_size -1 
                word = reverse_word_inde[num]
                word_embedding = weights[num]
                embedding_Matrix[word] = word_embedding 
            return embedding_Matrix
            

if __name__ =="__main__":
    work = sentiment_classification()
    work.load_data('../chinese_data/waimai_10k.csv')
    work.balanced_samples() 
    work.run()
