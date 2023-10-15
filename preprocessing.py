import os
import re
import zipfile
import json
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

ALL_TAGS = ['agobot', 'allaple', 'allowinfirewall', 'almanahe', 'amadey', 'amrat', 'antianalysis', 'antiav', 'antivm', 'arkeistealer', 'asackytrojan', 'asyncrat', 'autoitrunpe', 'avkeylogger', 'badpath', 'balrok', 'banker', 'base64encoded', 'berbew', 'bifrose', 'blackmoon', 'blackshades', 'bladabindi', 'blankgrabber', 'bobax', 'bodelph', 'brontok', 'buterat', 'ceatrg', 'cobaltstrike', 'coinminer', 'coins', 'coinstealer', 'crackedbyximo', 'crat', 'crealstealer', 'cybergate', 'cybergateinjector', 'darkcomet', 'dcrat', 'delphipacker', 'disabler', 'discordtokengrabber', 'dllinjectionshellcode', 'dorkbot', 'downloader', 'drat', 'drolnux', 'dropper', 'eggnog', 'fakeav', 'fakeextension', 'fakesvchost', 'fakesystemfile', 'farfli', 'fasong', 'floxif', 'formbook', 'ftpstealer', 'futurax', 'gandcrab', 'ganelp', 'genericantianalysis', 'genericantivm', 'genericdiscordtokengrabber', 'genericfakewindowsfile', 'generickeylogger', 'genericp2pworm', 'genericransomware', 'genericstealer', 'gepys', 'ghost', 'gigex', 'gleamal', 'heavensgate', 'hiveanalysis', 'hostsdisabler', 'imperium', 'keydoor', 'keylogger', 'killmbr', 'knightlogger', 'kryptik', 'lightmoon', 'loader', 'lokibot', 'lucifertokengrabber', 'ludbaruma', 'lummastealer', 'lunagrabber', 'lydra', 'maldosheader', 'mewsspy', 'mira', 'moarider', 'mofksys', 'moonlight', 'mumador', 'mydoom', 'nancatsurveillanceex', 'nanocore', 'neshta', 'netinjector', 'netkryptik', 'netlogger', 'netwire', 'nevereg', 'nitol', 'nofear', 'nworm', 'obfuscatedpersistence', 'obfuscatedrunpe', 'palevo', 'pastebinloader', 'picsys', 'pistolar', 'pluto', 'pony', 'prepscram', 'pupstudio', 'qbot', 'quasarrat', 'ramnit', 'ransomware', 'redline', 'rednet', 'reflectiveloader', 'remcosrat', 'revengerat', 'reverseencoded', 'ridnu', 'rifle', 'rotinom', 'rozena', 'runpe', 'sakularat', 'salgorea', 'sality', 'scar', 'sfone', 'shade', 'shellcode', 'shifu', 'sillyp2p', 'simbot', 'simda', 'smokeloader', 'snakekeylogger', 'socks', 'soltern', 'spy', 'stealer', 'stopransomware', 'suspiciouspdbpath', 'syla', 'telegrambot', 'tempedreve', 'tinba', 'tofsee', 'uacbypass', 'unruy', 'upatre', 'urelas', 'vbautorunworm', 'vbinjector', 'vboxvulndriver', 'vbsdropper', 'vflooder', 'vilsel', 'virut', 'vmprotect', 'vobfus', 'vulnerabledriver', 'wabot', 'windefenderdisabler', 'windex', 'winpayloads', 'wormautorun', 'wormyfier', 'xorencrypted', 'xredrat', 'xwormrat', 'zegost', 'zeus', 'zxshell']
ALL_TAGS_DICT = {tag: idx for idx, tag in enumerate(ALL_TAGS)}

def tags_to_vec(tags):
    vec = [0]*len(ALL_TAGS)
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        vec[ALL_TAGS_DICT[tag]] = 1
    return np.array(vec)

def vec_to_tags(vec):
    tags = []
    for idx, val in enumerate(vec):
        if val == 1:
            tags.append(ALL_TAGS[idx])
    return tags

def sequence_log(idx, total, freq=10):
    if idx%freq == 0:
        print(f'({idx}/{total})')

def extract_api_calls(source_dir, target_dir):
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
    os.remove('temp/apiTracing.txt')
    os.remove('temp/inMemoryResults.xml')   
    
def get_hash_tag_dict(filename):
    allowed_hashes = set([name.split('.')[0] for name in os.listdir('api-traces-smartvmi/dataset')])
    hash_tag_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
        tags = [tag for tag in data]
        for tag in tags:
            for hash in data[tag]:
                if hash in allowed_hashes:
                    if hash not in hash_tag_dict:
                        hash_tag_dict[hash] = set()
                    hash_tag_dict[hash].add(tag)
    for key, value in hash_tag_dict.items():
        hash_tag_dict[key] = list(value)
    return hash_tag_dict

def write_tags_to_json():
    hash_tag_dict = get_hash_tag_dict('api-traces-smartvmi/shas_by_families.json')
    with open('api-traces-smartvmi/families_by_shas.json', 'w') as f:
        json.dump(hash_tag_dict, f, indent=2)
    
def get_tags(hashes):
    with open('api-traces-smartvmi/families_by_shas.json', 'r') as f:
        data = json.load(f)
        tags = [data[hash] for hash in hashes]
    return tags
    
def find_hashes(tags, only=False):
    base = deepcopy(hashes)
    for tag in tags:
        base.intersection_update(tag_to_hashs[tag])
    if only:
        base = set(hash for hash in base if len(hash_to_tags[hash]) == len(tags))
    return base
    
# extract_api_calls('api-traces-smartvmi/dataset_raw', 'api-traces-smartvmi/dataset')
# write_tags_to_json()

hashes = set([name.split('.')[0] for name in os.listdir('api-traces-smartvmi/dataset')])

tag_to_hashs = {}
with open('api-traces-smartvmi/shas_by_families.json', 'r') as f:
    tag_to_hashs = json.load(f)
tag_to_hashs = {tag: set(hash) for tag, hash in tag_to_hashs.items()}
    
hash_to_tags = {}
with open('api-traces-smartvmi/families_by_shas.json', 'r') as f:
        hash_to_tags = json.load(f)

hash_to_tags_str = {hash: sorted(tags).__str__() for hash, tags in hash_to_tags.items()}
str_tag_dist = {}
for tag in hash_to_tags_str.values():
    if tag not in str_tag_dist:
        str_tag_dist[tag] = 0
    str_tag_dist[tag] += 1 

# total = 0
# tag_groups = []
# for tags, count in sorted(str_tag_dist.items(), key=lambda x: x[1], reverse=True):
#     if count < 100:
#         continue
#     print(count, tags)
#     print(len(find_hashes(tags[2:-2].split('\', \''))))
#     total += count
#     print('----------------')
# print(total)
# exit(0)

print('getting tags')
all_tags = [hash_to_tags[hash] for hash in hashes]
tag_vecs = np.array([tags_to_vec(tags) for tags in all_tags])

# print('fitting pca')
# pca = PCA(n_components=2)
# low_vec = pca.fit_transform(tag_vecs)

print('fitting clustering model')
k = 16
clustering_model = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
labels = clustering_model.fit_predict(tag_vecs)

for l in range(k):
    present_tags = {}
    for vec in tag_vecs[labels == l]:
        tags = vec_to_tags(vec)
        for tag in tags:
            if tag not in present_tags.keys():
                present_tags[tag] = 0
            present_tags[tag] += 1
    sorted_tags = sorted(present_tags.items(), key=lambda x: x[1], reverse=True)  # np.sum(labels == l)
    important_tags = []
    c = 0
    for tag, count in sorted_tags:
        if count > 0.9*np.sum(labels == l):
            c = count
            important_tags.append(tag)
    print(important_tags, c, np.sum(labels == l))
    print(sorted_tags)
    print(len(find_hashes(important_tags)))
    print('----------------------------')

# for l in range(k):
#     plt.plot(low_vec[l == labels, 0], low_vec[l == labels, 1], 'o')
# plt.show()


