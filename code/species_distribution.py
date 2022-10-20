import json
from pprint import pprint

import pandas
import json

json_data = None

with open('../data/SnapshotSerengeti_S1-11_v2.1.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

from collections import Counter

annotations = Counter()

categories = {}

for c in json_data['categories']:
    categories[c['id']] = c['name']

for annot in json_data['annotations']:
    annotations[categories[annot['category_id']]] += 1

#print(annotations)
#print('categories in season 1', len(annotations.keys()))
#print('n images', sum(annotations.values()))

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict(annotations, orient='index')#.reset_index()
print(df.columns)
df = df.sort_values(by=[0])
print(df.head())
bar = df.plot.bar(logy=True, legend=False, title='log-scale bar chart of species distribution\n in the Serengeti Snapshots dataset.').get_figure()
plt.tight_layout()
bar.savefig('../plots/species_distribution.png')