import os
import json
from sklearn.model_selection import train_test_split

def sequence_log(idx, total, freq=10):
    if idx%freq == 0:
        print(f'({idx}/{total})')  

def basic_split(source_folder, split_name='basic_split', train_size=0.8):
    filenames = os.listdir(source_folder)
    hashes = [name.split('.')[0] for name in filenames]
    train_files, test_files = train_test_split(hashes, train_size=train_size)
    # train_files, test_files = train_test_split(train_files, train_size=train_size)
    data = {'train':      train_files,
            'test':       test_files,
            # 'validation': validation_files
            }
    with open(f'data/{split_name}.json', 'w') as f:
        json.dump(data, f)



# return train test data and tags from split-file    
def get_train_test(split_name):
    with open(f'data/{split_name}.json', 'r') as f:
        mapping = json.load(f)
    train_hashes = set(mapping['train'])
    test_hashes = set(mapping['test'])
    
    print('loading data')
    with open(f'data/data_by_hash.json', 'r') as f:
        data_by_hash = json.load(f)  

    print('extracting train data')
    train_data = []
    train_tags = []
    for idx,hash in enumerate(train_hashes):
        # sequence_log(idx, len(train_hashes), 1000)
        data = data_by_hash.pop(hash)
        train_data.append(data['api_calls'])
        train_tags.append(data['tags'])
        
    print('extracting test data')
    test_data = []
    test_tags = []
    for idx,hash in enumerate(test_hashes):
        # sequence_log(idx, len(test_hashes), 1000)
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
    basic_split('data/dataset')
    # train_data, test_data, train_tags, test_tags = get_train_test('basic_split')
