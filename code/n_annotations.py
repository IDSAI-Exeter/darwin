import json

json_data = None

with open('../data/SnapshotSerengeti_S1-11_v2.1.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

print('# annotations with bbox', len([ d for d in json_data['annotations'] if 'bbox' in d.keys()]))