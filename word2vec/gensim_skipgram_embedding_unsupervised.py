
# 通过调用gensim.models.word2vec 来完成word embedding 

from  gensim.models import word2vec 
import os
import sys
import jieba 
import logging 
import gensim 
class  Gensim_embedding:
    def __init__(self,data_path):
        self.data_raw  =  {}
         #ignore_ = [',','?','-','=','\"','\'','<<','>>','...','。',':','!','!','(',')']
        id = 0 
        for line in open(data_path,'r',encoding = 'utf-8'):
            lines = line.strip().split(',',1)
            if lines[0] == 'label' : continue
            self.data_raw[id] = lines[1] 
            id += 1  

    def cut(self):
        data_cut={} 
        all_data = []
        ignore_flag = ['.........','（','"',"（",',','?','-','=','\"','\'','<<','>>','...','。',':','!','!','(',')']
        ignore_word = ['我','我们','他','她','如','如果','着','喔','的','还']
        for row in self.data_raw:
            rows = jieba.cut(self.data_raw[row],cut_all=False)
            filter_word = []             
            for field in list(rows):
                if field in ignore_flag or field in ignore_word: continue
                filter_word.append(field)
                all_data += filter_word
            if len(filter_word) < 200: continue 
            data_cut[row] = filter_word     
        return data_cut ,list(set(all_data ))  

    def run(self):
        save_model_file = 'fudan_embedding'
        self.id_data = {}
        self.id_data,all_data  = self.cut()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
        model = gensim.models.Word2Vec([all_data], min_count = 1 , size=4)  # 训练skip-gram模型; 默认window=5 , embedding_dimensions and hidden layers numbers  = 4  
        model.save(save_model_file)
        model.wv.save_word2vec_format(save_model_file + ".bin", binary=True)   # 以二进制类型保存模型以便重用
	   	
    def lookup(self):
        model_1 = word2vec.Word2Vec.load('fudan_embedding')
        y2 = model_1.most_similar("举目无亲", topn=10)
        print(y2)
		#for i in self.id_data:
        #    words = self.id_data[i]
       #    for j in words:
        #print(model_1[j])
if __name__ == "__main__":
    work = Gensim_embedding('../chinese_data/ChnSentiCorp_htl_all.csv')
    work.run()
    work.lookup()
