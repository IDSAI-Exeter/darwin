# Purpose : generate a file with bboxes and species for futher use and not have to load the whole dataset

import json
from pprint import pprint

import pandas
import json

json_data = None
#with open('../data/SnapshotSerengetiS01.json') as json_file:
with open('../data/SnapshotSerengeti_S1-11_v2.1.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

print('loaded annotations')

categories = {}

for c in json_data['categories']:
    categories[c['id']] = c['name']

images_species = {}

various_species = []

images = {}

for im in json_data['images']:
    images[im['id']] = im

annotations = {}

for annot in json_data['annotations']:
    if annot['image_id'] in images_species.keys() and categories[annot['category_id']] != images_species[annot['image_id']]:
        various_species.append(annot['image_id'])
    images_species[annot['image_id']] = categories[annot['category_id']]
    annotations[annot['image_id']] = annot

del json_data

print('#images with multiple species', len(various_species))
bbox_json_data = None

with open('../data/SnapshotSerengetiBboxes_20190903.json') as json_file:
    bbox_json_data = json.load(json_file)
    json_file.close()

print('loaded bbox')
bbox_species = []

for annot in bbox_json_data['annotations']:
    if annot['category_id'] == 1 and annot['image_id'] in images_species.keys() and annot['image_id'] not in various_species:# and annot['image_id'][0:2] == 'S1':
        bbox_species.append({
            'species': images_species[annot['image_id']],
            'image_id': annot['image_id'],
            'bbox': annot['bbox'],
            'annotation': annotations[annot['image_id']],
            'image': images[annot['image_id']]
        })

print('joined')
del bbox_json_data

json.dump(bbox_species, open('../data/bbox_species.json', 'w'))
