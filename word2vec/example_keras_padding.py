#!/bin/env/python
# -*- encoding:utf-8 -*-
#/**********************************************************
#* Author        : 07zhiping 
#* Email         : 07zhiping@gmail.com 
#* Last modified : 2020-11-01 10:36
#* Filename      : example_keras.padding.py
#* Description   : padding的使用，在training data,testdata 上的应用
#* *******************************************************/





import tensorflow as tt 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


somestr = ['ha ha gua angry','howa ha gua excited naive']
tok = Tokenizer(num_words=10,oov_token='<OOV>')
tok.fit_on_texts(somestr)
tok.word_index
print(tok.word_index)
#Out[90]: {'angry': 3, 'excited': 5, 'gua': 2, 'ha': 1, 'howa': 4, 'naive': 6}
print('traiining', tok.texts_to_sequences(somestr))
tok_sentence = tok.texts_to_sequences(somestr)
train_pad= pad_sequences(tok_sentence,maxlen=4,padding = 'post',truncating = 'post')
print('pad_training',train_pad)
# [[1, 1, 2, 3], [4, 1, 2, 5, 6]]

print(tok.word_index,'word_index')

print([(index,word) for (word,index) in tok.word_index.items()])
print('dict: ', dict([(index,word) for (word,index) in tok.word_index.items()]))

#print('padded',padded)

test = ['gua','gua','gua']
test2 = tok.texts_to_sequences(test)
print('test',test2)

pad_test = pad_sequences(test2,maxlen=4,padding='post',truncating='pre')
print('pad_test',pad_test)


print('paded',padded)
