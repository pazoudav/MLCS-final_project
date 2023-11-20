from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from copy import deepcopy
import json
import numpy as np
import os
from joblib import dump, load
from pprint import pprint

ALL_TAGS = ['agobot', 'allaple', 'allowinfirewall', 'almanahe', 'amadey', 'amrat', 'antianalysis', 'antiav', 'antivm', 'arkeistealer', 'asackytrojan', 'asyncrat', 'autoitrunpe', 'avkeylogger', 'badpath', 'balrok', 'banker', 'base64encoded', 'berbew', 'bifrose', 'blackmoon', 'blackshades', 'bladabindi', 'blankgrabber', 'bobax', 'bodelph', 'brontok', 'buterat', 'ceatrg', 'cobaltstrike', 'coinminer', 'coins', 'coinstealer', 'crackedbyximo', 'crat', 'crealstealer', 'cybergate', 'cybergateinjector', 'darkcomet', 'dcrat', 'delphipacker', 'disabler', 'discordtokengrabber', 'dllinjectionshellcode', 'dorkbot', 'downloader', 'drat', 'drolnux', 'dropper', 'eggnog', 'fakeav', 'fakeextension', 'fakesvchost', 'fakesystemfile', 'farfli', 'fasong', 'floxif', 'formbook', 'ftpstealer', 'futurax', 'gandcrab', 'ganelp', 'genericantianalysis', 'genericantivm', 'genericdiscordtokengrabber', 'genericfakewindowsfile', 'generickeylogger', 'genericp2pworm', 'genericransomware', 'genericstealer', 'gepys', 'ghost', 'gigex', 'gleamal', 'heavensgate', 'hiveanalysis', 'hostsdisabler', 'imperium', 'keydoor', 'keylogger', 'killmbr', 'knightlogger', 'kryptik', 'lightmoon', 'loader', 'lokibot', 'lucifertokengrabber', 'ludbaruma', 'lummastealer', 'lunagrabber', 'lydra', 'maldosheader', 'mewsspy', 'mira', 'moarider', 'mofksys', 'moonlight', 'mumador', 'mydoom', 'nancatsurveillanceex', 'nanocore', 'neshta', 'netinjector', 'netkryptik', 'netlogger', 'netwire', 'nevereg', 'nitol', 'nofear', 'nworm', 'obfuscatedpersistence', 'obfuscatedrunpe', 'palevo', 'pastebinloader', 'picsys', 'pistolar', 'pluto', 'pony', 'prepscram', 'pupstudio', 'qbot', 'quasarrat', 'ramnit', 'ransomware', 'redline', 'rednet', 'reflectiveloader', 'remcosrat', 'revengerat', 'reverseencoded', 'ridnu', 'rifle', 'rotinom', 'rozena', 'runpe', 'sakularat', 'salgorea', 'sality', 'scar', 'sfone', 'shade', 'shellcode', 'shifu', 'sillyp2p', 'simbot', 'simda', 'smokeloader', 'snakekeylogger', 'socks', 'soltern', 'spy', 'stealer', 'stopransomware', 'suspiciouspdbpath', 'syla', 'telegrambot', 'tempedreve', 'tinba', 'tofsee', 'uacbypass', 'unruy', 'upatre', 'urelas', 'vbautorunworm', 'vbinjector', 'vboxvulndriver', 'vbsdropper', 'vflooder', 'vilsel', 'virut', 'vmprotect', 'vobfus', 'vulnerabledriver', 'wabot', 'windefenderdisabler', 'windex', 'winpayloads', 'wormautorun', 'wormyfier', 'xorencrypted', 'xredrat', 'xwormrat', 'zegost', 'zeus', 'zxshell']
ALL_TAGS_DICT = {tag: idx for idx, tag in enumerate(ALL_TAGS)}



# labbeler is a class which assingns groups of tags a single label
# as a result all malware traces have only single lable, not multiple tags, and can be used with standart classifiers
# Labeller is a base class for different labelling aproaches
# any labeller can be saved and loaded from file
class Labeller():
    def __init__(self) -> None:
        self.name = 'Basic'
        self.hashes = set()
        self.tag_to_hashs = {}
        self.hash_to_tags = {}
        
    def make(self):
        self.hashes = set([name.split('.')[0] for name in os.listdir('data/dataset')])

        self.tag_to_hashs = {}
        with open('data/shas_by_families.json', 'r') as f:
            self.tag_to_hashs = json.load(f)
        self.tag_to_hashs = {tag: set(hash) for tag, hash in self.tag_to_hashs.items()}
            
        self.hash_to_tags = {}
        with open('data/families_by_shas.json', 'r') as f:
                self.hash_to_tags = json.load(f)
        return self                 
                           
    def save(self):
        dump(self, f'labellers/{self.name}.lbl')
        return self
    
    def load(self):
        self : __class__ = load(f'labellers/{self.name}.lbl')
        return self
                
    def vec_to_tags(self, vec):
        tags = []
        for idx, val in enumerate(vec):
            if val == 1:
                tags.append(ALL_TAGS[idx])
        return tags
    
    def tags_to_vec(self, tags):
        vec = [0]*len(ALL_TAGS)
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            vec[ALL_TAGS_DICT[tag]] = 1
        return np.array(vec)

    def get_tags(self, hashes):
        with open('data/families_by_shas.json', 'r') as f:
            data = json.load(f)
            tags = [data[hash] for hash in hashes]
        return tags
        
    def find_hashes(self, tags, only=False):
        base = deepcopy(self.hashes)
        for tag in tags:
            base.intersection_update(self.tag_to_hashs[tag])
        if only:
            base = set(hash for hash in base if len(self.hash_to_tags[hash]) == len(tags))
        return base
        
    def label(self, tags_list):
        return [self.tags_to_vec(tags) for tags in tags_list]
   
   
# simply numbers all unique combinations of labels thar occur in the dataset
class EnumerativeLabeller(Labeller):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Enumerative'
        self.vec_to_num = {}
    
    def make(self):
        super().make() 
        unique_vecs = set(tuple(self.tags_to_vec(tags)) for tags in self.hash_to_tags.values())
        self.vec_to_num = {vec: idx for idx,vec in enumerate(unique_vecs)}        
        return self 
    
    def label(self, tags_list):
        labels = []
        for tags in tags_list:
            labels.append(self.vec_to_num[tuple(self.tags_to_vec(tags))])
        return labels
 
 
# clusters tags based on the K-means algorithm
# K is a user defined parameter   
# will probaly not be used...  
class KMeansLabeller(Labeller):
    def __init__(self, k) -> None:
        super().__init__()
        self.name = f'KMeans{k}'
        self.k = k
        
    def make(self):
        super().make()
        all_tags = [self.hash_to_tags[hash] for hash in self.hashes]
        self.tag_vecs = np.array([self.tags_to_vec(tags) for tags in all_tags])
        self.model = KMeans(n_clusters=self.k, init='k-means++', n_init='auto', max_iter=100000)
        self.model.fit(self.tag_vecs)
        return self
    
    def label(self, tags_list):
        tags_list = [self.tags_to_vec(tags) for tags in tags_list]
        return self.model.predict(tags_list)
    
    def analyze(self, labels):
        all_important_tags = []
        for l in range(self.k):
            present_tags = {}
            for vec in self.tag_vecs[labels == l]:
                tags = self.vec_to_tags(vec)
                for tag in tags:
                    if tag not in present_tags.keys():
                        present_tags[tag] = 0
                    present_tags[tag] += 1
            sorted_tags = sorted(present_tags.items(), key=lambda x: x[1], reverse=True) 
            important_tags = []
            c = 0
            for tag, count in sorted_tags:
                if count > 0.9*np.sum(labels == l):
                    c = count
                    important_tags.append(tag)
            important_tags = sorted(important_tags)
            all_important_tags.append([important_tags, c, np.sum(labels == l),len(self.find_hashes(important_tags))])
            print(sorted(important_tags), c, np.sum(labels == l))
            print(sorted_tags)
            print(len(self.find_hashes(important_tags)))
            print('----------------------------')
        return all_important_tags
   
   
# takes the 'k' most common tag groupings until 'ratio' of all traces
# don't have a good explanation, may be simpler to show in pseudocode     
# my prefered labelling method
class SubsetLabeller(Labeller):
    def __init__(self, ratio=0.8) -> None:
        super().__init__()
        self.name = f'Subset{int(ratio*100)}'
        self.subsets = []
        self.ratio = ratio
        
    def make(self):
        super().make()
        tags = self.get_tags(self.hashes)
        tags = [tuple(sorted(x)) for x in tags]
        unique_tags = list(set(tags))
        unique_tags_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        stats = [[0,t] for t in unique_tags]
        for tag in tags:
            stats[unique_tags_idx[tag]][0] += 1
        stats = sorted(stats, key=lambda x: x[0], reverse=True)
        count = 0    
        large_tags = []
        for s in stats:
            count += s[0]
            large_tags.append(s[1])
            if count >= len(tags)*self.ratio:
                break
        large_tags = sorted(large_tags, key=lambda x: len(x), reverse=True)
        self.subsets = [set(t) for t in large_tags] + [set()]
        return self

    def label(self, tags_list):
        labels = []
        for tags in tags_list:
            tags_set = set(tags)
            for idx, subset in enumerate(self.subsets):
                if len(subset.intersection(tags_set)) == len(subset):
                    labels.append(idx)
                    break
        return labels  
    
     
if __name__ == '__main__':
    ...
    # s = SubsetLabeller().load()
    # all_tags = s.get_tags(s.hashes)
    # all_labels = s.label(all_tags)
    # hystogram = [0]*len(s.subsets)
    # for x in all_labels:
    #     hystogram[x] += 1
    # print(list(zip(hystogram, s.subsets)))
    # # print(len(s))
    