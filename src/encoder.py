from split_data import get_train_test
from joblib import dump, load
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


# encoder creates a vector of fixed length of features extracted from the malware's API calls
# Encoder is a base class for more complicated encoders
# any encoder can be saved and loaded from file
class Encoder():
    def __init__(self, name_flag='') -> None:
        self.name = f'Basic{name_flag}'
    
    def make(self, split='basic_split'):
        return self
    
    def encode(self, data):
        return data
    
    def save(self):
        dump(self, f'encoders/{self.name}.enc')
        return self
    
    def load(self):
        self : __class__ = load(f'encoders/{self.name}.enc')
        return self
    
# simple bag of words 
class BagOfWordsEncoder(Encoder):
    def __init__(self, name_flag='') -> None:
        super().__init__()
        self.name = f'BagOfWords{name_flag}'
        self.encoder = None
        self.API_calls = []
        self.API_calls_set = set()
        self.API_calls_dict = {}   
        
    def make(self, split='basic_split'):
        X_train, _, _, _ = get_train_test(split)
        for sample in X_train:
            for API_call in sample:
                self.API_calls_set.add(API_call)
        self.API_calls_set.add('UNK')
        self.API_calls = list(self.API_calls_set)
        self.API_calls_dict = {API_call: idx for idx, API_call in enumerate(self.API_calls)}
        return self
        
    def encode(self, data):
        transformed_data = []
        for sample in data:
            transformed_sample = [0]*len(self.API_calls)
            for API_call in sample:
                if API_call not in self.API_calls_set:
                    API_call = 'UNK'
                transformed_sample[self.API_calls_dict[API_call]] = 1
            transformed_data.append(transformed_sample)
        return transformed_data
      
# extension of BagOfWordsEncoder, counts the number of times API call occurs in the malware trace  
# probably should be called 'CountEncoder'
class FrequencyEncoder(BagOfWordsEncoder):
    def __init__(self, name_flag='') -> None:
        super().__init__()
        self.name = f'Frequency{name_flag}'
        
    def encode(self, data):
        transformed_data = []
        for sample in data:
            transformed_sample = [0]*len(self.API_calls)
            for API_call in sample:
                if API_call not in self.API_calls_set:
                    API_call = 'UNK'
                transformed_sample[self.API_calls_dict[API_call]] += 1
            transformed_data.append(transformed_sample)
        return transformed_data
    

# creates N-grams of the malware traces
# N can be set by user
# this encoding takes into account order in which API calls are called
class NGramEncoder(Encoder):
    def __init__(self,n, name_flag='') -> None:
        super().__init__()
        self.name = f'{n}Gram{name_flag}'
        self.n = n
        self.API_gram_set = set()
        self.API_gram_dict = {}
        
    def make(self, split='basic_split'):
        X_train, _, _, _ = get_train_test(split)
        for idx, sample in enumerate(X_train):
            if idx % 1000 == 0:
                print(idx)
            for API_gram in list(ngrams(sample, self.n)):
                self.API_gram_set.add(API_gram)
        self.API_gram_set.add('UNK')
        self.API_gram_dict = {API_gram: idx for idx, API_gram in enumerate(self.API_gram_set)}
        return self
        
    def encode(self, data):
        transformed_data = np.zeros((len(data), len(self.API_gram_set)), dtype=np.uint8)
        for id, sample in enumerate(data):
            if id%1000 == 0:
                print(id)
            # transformed_sample = [False]*len(self.API_gram_set)
            API_grams = list(ngrams(sample, self.n))
            for API_gram in API_grams:
                if API_gram not in self.API_gram_set:
                    API_gram = 'UNK'
                idx = self.API_gram_dict[API_gram]
                transformed_data[id,idx] = 1
            # transformed_data.append(np.array(transformed_sample, dtype=np.uint8))
        return transformed_data


# Term Frequency - Inverse Document Frequency (TF-IDF)
# common in language processing for encoding of documents
# takes into account relative frequencies of call in the malware trace and across all malware traces
class TF_IDFEncoder(Encoder):
    def __init__(self, name_flag='') -> None:
        super().__init__()
        self.name = f'TF-IDF{name_flag}'
        self.freq_enc = FrequencyEncoder()
        self.transformer = TfidfTransformer()
        
    def make(self, split='basic_split'):
        super().make(split)
        self.freq_enc.make(split)
        data, _, _, _ = get_train_test(split)
        data = self.freq_enc.encode(data)
        self.transformer.fit(data)
        return self
    
    def encode(self, data):
        data_ = self.freq_enc.encode(data)
        data_ = self.transformer.transform(data_).toarray()
        return np.array(data_)
    
    
# doesn't create a vector of fixed length, maybe useful for RNN 
# probably won't be used 
class Numeric(BagOfWordsEncoder):
    def __init__(self, name_flag='') -> None:
        super().__init__(name_flag)
        self.name = f'Numeric{name_flag}'
    
    def encode(self, data):
        transformed_data = []
        for sample in data:
            transformed_sample = []
            for API_call in sample:
                if API_call not in self.API_calls_set:
                    API_call = 'UNK'
                transformed_sample.append(self.API_calls_dict[API_call])
            transformed_data.append(np.array(transformed_sample))
        return transformed_data
    
# creates 2d one hot encoding of malware trace up to length 4096, 
# used in NN 
class OneHotEncoder(Numeric):
    def __init__(self, max_len=4096 ,name_flag='') -> None:
        super().__init__(name_flag)
        self.name = f'OneHot-{max_len}{name_flag}'
        self.max_len = max_len
        self.call_count = 0
        
    def make(self, split='basic_split'):
        self = super().make(split)
        self.call_count = len(self.API_calls) + 1
        return self
    
    def encode(self, data):
        transformed_data_ = super().encode(data)
        transformed_data = np.zeros((len(data), self.max_len, self.call_count,1),dtype=np.uint8)
        # transformed_data = np.full((len(data),self.max_len), self.call_count-1)
        for idx, trace in enumerate(transformed_data_):
            if idx%1000 == 0:
                print(idx)
            if len(trace) > self.max_len:
                trace = trace[:self.max_len]
            transformed_data[idx,range(len(trace)),trace,0] = 1 
            # rng = min(self.max_len, len(trace))
            # transformed_data[idx,:rng] = trace[:rng]     

        return transformed_data
            
                
    

