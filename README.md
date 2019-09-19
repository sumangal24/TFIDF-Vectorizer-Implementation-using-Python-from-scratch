# TFIDF-Vectorizer-Implementation-using-Python-from-scratch
#Instead of using TFIDF vectorizer , sometimes for very large datasets we have to write the code from scratch due to memory issues.
#𝐼𝐷𝐹(𝑡)=1+log𝑒(1 + Total number of documents in collection)/(1+Number of documents with term t in it)
#𝑇𝐹(𝑡)=Number of times term t appears in a document/Total number of terms in the document



from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


# Using the same given in references.
#function for fit for tfidf vector


def fit(corpus):
    """
    This function gives all unique words in a particular document
    """
    unique_words = set()
    idf_list = []
    if isinstance(corpus,(list)):
        for row in corpus:
            for word in row.split(" "):                
                if len(word) < 2:
                    continue
                unique_words.add(word)
    
        unique_words = sorted(list(unique_words))
        vocab = {j:i for i,j in enumerate(unique_words)}
        #print(vocab)
        doclist = []
        for row in (tqdm(corpus)):
            dictA = dict.fromkeys(vocab,0)
            for word in row.split(" "):
                if len(word)<2:
                    continue
                dictA[word] +=1
            doclist.append(dictA)
        idf = compute_IDF(doclist)
    #print(idf)
        #idf = idf_value.values()
        #idf_list.append(idf_value.values())
    return vocab,idf
    
    
    def compute_IDF(doclist): 
    """
    This function calculates the value of idf
    """
    idf_dict = {}
    idf_list = []
    N = len(doclist)
    #print(N)
    idf_dict = dict.fromkeys(doclist[0].keys(),0) #comparing bag of words with our documents to get the frequency of unique doc in corpus
    #print(idf_dict)
    for doc in doclist:
        for word,val in doc.items():
            if val>0:
                idf_dict[word] +=1
    #print(idf_dict)    
    for word,val in idf_dict.items():
        #print(word)
        #print(val)
        idf_dict[word] = 1 + math.log((N+1)/(val+1))
        
    return idf_dict
    
    def transform(corpus,vocab):
    """
    This function will print the normalized sparse matrix of tfidf
    """
    rows = []
    columns = []
    tf_values = []
    tf_dict = []
    tf_idf=[]
    if isinstance(corpus, (list)):
        for idx, row in enumerate(tqdm(corpus)): # for each document in the dataset  
            tfdict = {}
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row.split()))
            # for every unique word in the document
            for word, freq in word_freq.items():  # for each unique word in the review.                
                if len(word) < 2:
                    continue
                col_index = vocab.get(word, -1) # retreving the dimension number of a word
                # if the word exists
                if col_index !=-1:
                    # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)
                    tf = word_freq[word]/sum(word_freq.values())
                    tfdict[word] = word_freq[word]/sum(word_freq.values())
                tf_values.append(tf)
            #print(tf_values)
            tf_dict.append(tfdict) 
        #print(tf_dict)
            
            

        for doc in tf_dict:
            tfidf = {}
            for word, val in doc.items():
                tfidf[word] = val*idf_val[word]
                #print(tfidf[word])
                tfidf_values = tfidf.values()
                tf_idf.append(tfidf[word])
            #print(tf_idf)
            
            #print(tfidf)
        
        #print("TF MATRIX IS :" ,csr_matrix((tf_values, (rows,columns)), shape=(len(corpus),len(vocab))))
        tfidf_matrix = csr_matrix((tf_idf, (rows,columns)), shape=(len(corpus),len(vocab)))
        #print(tfidf_matrix)
        #print("="*50)
        tfidf_mat_norm = normalize(tfidf_matrix)
        #print(tfidf_mat_norm)
        return tfidf_mat_norm
    else:
        print("you need to pass list of strings")


#This code will give you the same output as given by sklearn.

.
