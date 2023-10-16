from sklearn.feature_extraction.text import CountVectorizer
from split_data import get_train_test
from joblib import dump, load

class Encoder():
    def __init__(self) -> None:
        self.name = 'Basic'
    
    def encode(self, data):
        return data
    
    def save(self):
        dump(self, f'encoders/{self.name}.enc')
        return self
    
    def load(self):
        self = load(f'encoders/{self.name}.enc')
        return self
    
    
class BagOfWordsEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'BagOfWords'
        self.encoder = None
        self.API_calls = []
        self.API_calls_set = set()
        self.API_calls_dict = {}
        
        
    def make(self):
        X_train, _, _, _ = get_train_test('basic_split')
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
            transformed_sample = [False]*len(self.API_calls)
            for API_call in sample:
                if API_call not in self.API_calls_set:
                    API_call = 'UNK'
                transformed_sample[self.API_calls_dict[API_call]] = True
            transformed_data.append(transformed_sample)
        return transformed_data
        
class FrequencyEncoder(BagOfWordsEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Frequency'
        
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
    


