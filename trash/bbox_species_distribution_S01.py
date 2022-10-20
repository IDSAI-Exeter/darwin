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

print(json_data.keys())
print(json_data['info'])

from collections import Counter

annotations = Counter()

print(json_data['categories'][0])

categories = {}

for c in json_data['categories']:
    categories[c['id']] = c['name']

print(json_data['annotations'][0])

images_species = {}

for annot in json_data['annotations']:
    annotations[categories[annot['category_id']]] += 1
    images_species[annot['image_id']] = categories[annot['category_id']]

print(annotations)
print('categories in season 1', len(annotations.keys()))
print('n images', sum(annotations.values()))

bbox_json_data = None

with open('../data/SnapshotSerengetiBboxes_20190903.json') as json_file:
    bbox_json_data = json.load(json_file)
    json_file.close()

species_bbox = Counter()

for annot in bbox_json_data['annotations']:
    if annot['category_id'] == 1 and annot['image_id'][0:2] == 'S1':
        species_bbox[images_species[annot['image_id']]] += 1

print('species bbox', species_bbox)

print('n species bbox s01', len(species_bbox.keys()))

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict(species_bbox, orient='index')#.reset_index()
print(df.columns)
df = df.sort_values(by=[0])
print(df.head())
bar = df.plot.bar(logy=True, legend=False, title='log-scale bar chart of species distribution\n in the images with bounding boxes annotations\n in Season 1 of Serengeti dataset.').get_figure()
plt.tight_layout()
bar.savefig('s01_bbox.png')


