import os
import json
from sklearn.model_selection import train_test_split

def sequence_log(idx, total, freq=10):
    if idx%freq == 0:
        print(f'({idx}/{total})')  

# creates a train test split for the data and saves the hashes to a 'split-file'
# ensures consistency between training/test runs
# as a result same train data used every time
def basic_split(filename, split_name, train_size=0.8):
    with open(f'data/{filename}.json', 'r') as f:
        data = json.load(f)
    hashes = list(data.keys())
    train_files, test_files = train_test_split(hashes, train_size=train_size)
    data = {'source':     filename,
            'train':      train_files,
            'test':       test_files
            }
    with open(f'data/{split_name}.json', 'w') as f:
        json.dump(data, f)
  
# creates smaller train test split for the data and saves the hashes to a 'split-file'
# used for faster parameter tunning
def small_split(filename, split_name, higher_split_name):
    with open(f'data/{filename}.json', 'r') as f:
        data = json.load(f)
    with open(f'data/{higher_split_name}.json', 'r') as f:
        mapping = json.load(f)
        train_hashes = set(mapping['train'])
    hashes = list(data.keys())
    train_files = []
    test_files = []
    for hash in hashes:
        if hash in train_hashes:
            train_files.append(hash)
        else:
            test_files.append(hash)
    data = {'source':     filename,
            'train':      train_files,
            'test':       test_files  
            }
    with open(f'data/{split_name}.json', 'w') as f:
        json.dump(data, f)



# return train test data and tags from split-file    
def get_train_test(split_name):
    with open(f'data/{split_name}.json', 'r') as f:
        mapping = json.load(f)
    train_hashes = mapping['train'] 
    test_hashes = mapping['test']
    source_file = mapping['source']
    
    print('loading data')
    with open(f'data/{source_file}.json', 'r') as f:
        data_by_hash = json.load(f)  

    print('extracting train data')
    train_data = []
    train_tags = []
    for hash in train_hashes:
        data = data_by_hash.pop(hash)
        train_data.append(data['api_calls'])
        train_tags.append(data['tags'])
        
    print('extracting test data')
    test_data = []
    test_tags = []
    for hash in test_hashes:
        data = data_by_hash.pop(hash)
        test_data.append(data['api_calls'])
        test_tags.append(data['tags']) 

    return train_data, test_data, train_tags, test_tags   
   
# return validation data and tags from split-file   
def get_validation(split_name):
    with open(f'data/{split_name}.json', 'r') as f:
        mapping = json.load(f)
    validation_hashes = set(mapping['validation'])
    
    with open(f'data/data_by_hash.json', 'r') as f:
        data_by_hash = json.load(f)  

    validation_data = []
    validation_tags = []
    for hash in validation_hashes:
        data = data_by_hash.pop(hash)
        validation_data.append(data['api_calls'])
        validation_tags.append(data['tags'])

    return validation_data, validation_tags   

if __name__ == '__main__':
    basic_split('data_by_hash', 'basic_split')
    small_split('data_by_hash-small', 'basic_split-small', 'basic_split')
    # train_data, test_data, train_tags, test_tags = get_train_test('basic_split')
