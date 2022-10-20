import json
from pprint import pprint

import pandas
import json

json_data = None

with open('../data/SnapshotSerengetiS01.json') as json_file:
    #df = pandas.read_json(json_file)
    json_data = json.load(json_file)
    json_file.close()
    #df.head()

from collections import Counter

annotations = Counter()

categories = {}

for c in json_data['categories']:
    categories[c['id']] = c['name']

images_species = {}

for annot in json_data['annotations']:
    annotations[categories[annot['category_id']]] += 1
    images_species[annot['image_id']] = categories[annot['category_id']]

bbox_json_data = None
with open('../data/SnapshotSerengetiBboxes_20190903.json') as json_file:
    bbox_json_data = json.load(json_file)
    json_file.close()

bbox_ids = []

for annot in bbox_json_data['annotations']:
    if annot['category_id'] == 1 and annot['image_id'][0:2] == 'S1':
    #if annot['image_id'][0:2] == 'S1':
        bbox_ids.append(annot['image_id'])

bbox_ids = list(set(bbox_ids))
import os
n = len(bbox_ids)
print('#imagesbbox', n)


batchsize = 100
from_ = 0

for i in range(from_, len(bbox_ids), batchsize):
    batch = bbox_ids[i:i+batchsize] 
    open("download_file_list.txt", 'w').write("\n".join(["%s.JPG"%name for name in batch]))
    os.system("../lib/azcopy/azcopy cp  https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped/ '../data/images/' --list-of-files download_file_list.txt --output-level quiet")
    print('#remainingimagesbbox', len(bbox_ids) - i-batchsize, '#next_i', i+batchsize)