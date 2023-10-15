import os
import re
import zipfile
import json
import numpy as np


def sequence_log(idx, total, freq=10):
    if idx%freq == 0:
        print(f'({idx}/{total})')

def extract_api_calls(source_dir, target_dir):
    os.mkdir('temp')
    filenames = os.listdir(source_dir)
    for idx, filename in enumerate(filenames):
        sequence_log(idx, len(filenames))
        api_call_vector = []
        filehash = filename.split('-')[0]
        with zipfile.ZipFile(f'{source_dir}/{filename}', 'r') as zip_ref:
            zip_ref.extractall('temp/')
        with open('temp/apiTracing.txt', 'r') as f:
            total = 0
            for line in f.readlines():
                total += 1
                api_call = re.findall('\"vmi_Function\":\"([a-zA-Z0-9]*)\"', line)
                if len(api_call) > 0:
                    api_call_vector.append(api_call[0] + ',')
        with open(f'{target_dir}/{filehash}.txt', 'w') as f:
            f.writelines(api_call_vector)
    os.remove('temp')
    

def make_families_by_shas_json():
    allowed_hashes = set([name.split('.')[0] for name in os.listdir('data/dataset')])
    hash_tag_dict = {}
    with open('data/shas_by_families.json', 'r') as f:
        data = json.load(f)
        tags = [tag for tag in data]
        for idx, tag in enumerate(tags):
            sequence_log(idx, len(tags))
            for hash in data[tag]:
                if hash in allowed_hashes:
                    if hash not in hash_tag_dict:
                        hash_tag_dict[hash] = set()
                    hash_tag_dict[hash].add(tag)
                    
    for key, value in hash_tag_dict.items():
        hash_tag_dict[key] = list(value)

    with open('data/families_by_shas.json', 'w') as f:
        json.dump(hash_tag_dict, f, indent=2)
    
    
if __name__ == '__main__':
    extract_api_calls('data/dataset_raw', 'data/dataset')
    make_families_by_shas_json()



