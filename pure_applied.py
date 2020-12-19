import pandas as pd
import numpy as np
import tensorflow as tf
import re


import unidecode #for removing accents from strings.



import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

from nltk.stem.porter import PorterStemmer



stemmer = PorterStemmer()
def get_string_tokens(string):

    string = re.sub("(\$.+?\$)","",string, flags = re.DOTALL) #Removes Latex
    string = remove_stopwords(string.lower()) #Removes Stopwords and lowers all characters
    string = unidecode.unidecode(string) #Replaces accents, e.g. ö --> o
    string = string.replace("\n"," ") #Replace newlines with just a space
    string = re.sub("(\\\\+.)","",string) #This removes some remaining latex often used for accents, like Carath\'eodory --> Caratheodory.
    string = re.sub("([\{\}]+)","",string) #Some accents are written  Carath\'{e}odory, which becomes Carath{e}odory by the previous step, so we remove the brackets.
    tokens = simple_preprocess(string, max_len=100) #Removes symbols and returns list of tokens
    tokens = [stemmer.stem(token) for token in tokens] #stems tokens
    
    return tokens


bigrammer = gensim.models.phrases.Phrases.load("bigram.model")

wv_model = gensim.models.Word2Vec.load("word2vec.model")



class get_embedding:
    def __init__(self,model,bigrammer):
        self.word2vec = model
        self.bigrammer = bigrammer 
        
        
    def __call__(self,text):
        tokens = get_string_tokens(text)
        bigrammed_text = bigrammer[tokens]
        in_vocab = set()
        for word in bigrammed_text:
            if word in self.word2vec.wv.vocab:
                in_vocab.add(word)
        if in_vocab:
            return np.mean([self.word2vec.wv[word] for word in in_vocab], axis=0)
        else:
            return np.zeros(self.word2vec.vector_size)
        
embedding = get_embedding(model=wv_model, bigrammer=bigrammer)

MSN_model = tf.keras.models.load_model('models/MSN_model.h5')

def predict(string, prob=False):
    vector = np.array([embedding(string)])
    pure =  MSN_model.predict(vector).flatten()[0]
    if prob==True:
        return pure
    else:
        return np.rint(pure)
    
    


