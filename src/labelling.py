from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from copy import deepcopy
import json
import numpy as np
import os

ALL_TAGS = ['agobot', 'allaple', 'allowinfirewall', 'almanahe', 'amadey', 'amrat', 'antianalysis', 'antiav', 'antivm', 'arkeistealer', 'asackytrojan', 'asyncrat', 'autoitrunpe', 'avkeylogger', 'badpath', 'balrok', 'banker', 'base64encoded', 'berbew', 'bifrose', 'blackmoon', 'blackshades', 'bladabindi', 'blankgrabber', 'bobax', 'bodelph', 'brontok', 'buterat', 'ceatrg', 'cobaltstrike', 'coinminer', 'coins', 'coinstealer', 'crackedbyximo', 'crat', 'crealstealer', 'cybergate', 'cybergateinjector', 'darkcomet', 'dcrat', 'delphipacker', 'disabler', 'discordtokengrabber', 'dllinjectionshellcode', 'dorkbot', 'downloader', 'drat', 'drolnux', 'dropper', 'eggnog', 'fakeav', 'fakeextension', 'fakesvchost', 'fakesystemfile', 'farfli', 'fasong', 'floxif', 'formbook', 'ftpstealer', 'futurax', 'gandcrab', 'ganelp', 'genericantianalysis', 'genericantivm', 'genericdiscordtokengrabber', 'genericfakewindowsfile', 'generickeylogger', 'genericp2pworm', 'genericransomware', 'genericstealer', 'gepys', 'ghost', 'gigex', 'gleamal', 'heavensgate', 'hiveanalysis', 'hostsdisabler', 'imperium', 'keydoor', 'keylogger', 'killmbr', 'knightlogger', 'kryptik', 'lightmoon', 'loader', 'lokibot', 'lucifertokengrabber', 'ludbaruma', 'lummastealer', 'lunagrabber', 'lydra', 'maldosheader', 'mewsspy', 'mira', 'moarider', 'mofksys', 'moonlight', 'mumador', 'mydoom', 'nancatsurveillanceex', 'nanocore', 'neshta', 'netinjector', 'netkryptik', 'netlogger', 'netwire', 'nevereg', 'nitol', 'nofear', 'nworm', 'obfuscatedpersistence', 'obfuscatedrunpe', 'palevo', 'pastebinloader', 'picsys', 'pistolar', 'pluto', 'pony', 'prepscram', 'pupstudio', 'qbot', 'quasarrat', 'ramnit', 'ransomware', 'redline', 'rednet', 'reflectiveloader', 'remcosrat', 'revengerat', 'reverseencoded', 'ridnu', 'rifle', 'rotinom', 'rozena', 'runpe', 'sakularat', 'salgorea', 'sality', 'scar', 'sfone', 'shade', 'shellcode', 'shifu', 'sillyp2p', 'simbot', 'simda', 'smokeloader', 'snakekeylogger', 'socks', 'soltern', 'spy', 'stealer', 'stopransomware', 'suspiciouspdbpath', 'syla', 'telegrambot', 'tempedreve', 'tinba', 'tofsee', 'uacbypass', 'unruy', 'upatre', 'urelas', 'vbautorunworm', 'vbinjector', 'vboxvulndriver', 'vbsdropper', 'vflooder', 'vilsel', 'virut', 'vmprotect', 'vobfus', 'vulnerabledriver', 'wabot', 'windefenderdisabler', 'windex', 'winpayloads', 'wormautorun', 'wormyfier', 'xorencrypted', 'xredrat', 'xwormrat', 'zegost', 'zeus', 'zxshell']
ALL_TAGS_DICT = {tag: idx for idx, tag in enumerate(ALL_TAGS)}


class Labeller():
    def __init__(self) -> None:
        self.hashes = set([name.split('.')[0] for name in os.listdir('data/dataset')])

        self.tag_to_hashs = {}
        with open('data/shas_by_families.json', 'r') as f:
            self.tag_to_hashs = json.load(f)
        self.tag_to_hashs = {tag: set(hash) for tag, hash in self.tag_to_hashs.items()}
            
        self.hash_to_tags = {}
        with open('data/families_by_shas.json', 'r') as f:
                self.hash_to_tags = json.load(f)
                
        print('getting tags')
        all_tags = [self.hash_to_tags[hash] for hash in self.hashes]
        self.tag_vecs = np.array([self.tags_to_vec(tags) for tags in all_tags])
                
    
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
   
   
class KMeansLabeller(Labeller):
    def __init__(self, k) -> None:
        super().__init__()
        print('fitting clustering model')
        self.k = 8
        self.model = KMeans(n_clusters=k, init='k-means++', n_init='auto', max_iter=10000)
        self.model.fit(self.tag_vecs)
    
    def label(self, tags_list):
        return self.model.predict(tags_list)
    
    def analyze(self, labels):
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
            print(important_tags, c, np.sum(labels == l))
            print(sorted_tags)
            print(len(self.find_hashes(important_tags)))
            print('----------------------------')
     
   



# hash_to_tags_str = {hash: sorted(tags).__str__() for hash, tags in hash_to_tags.items()}
# str_tag_dist = {}
# for tag in hash_to_tags_str.values():
#     if tag not in str_tag_dist:
#         str_tag_dist[tag] = 0
#     str_tag_dist[tag] += 1 

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

# print('fitting pca')
# pca = PCA(n_components=2)
# low_vec = pca.fit_transform(tag_vecs)


# for l in range(k):
#     plt.plot(low_vec[l == labels, 0], low_vec[l == labels, 1], 'o')
# plt.show()