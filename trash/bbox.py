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

for annot in json_data['annotations']:
    annotations[categories[annot['category_id']]] += 1

print(annotations)
print('categories in season 1', len(annotations.keys()))
print('n images', sum(annotations.values()))
