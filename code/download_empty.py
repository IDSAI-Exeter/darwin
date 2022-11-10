import json
import os

try:
    os.mkdir('../data/empty/')
except:
    pass


json_data = None

# with open('../data/bbox_species.json') as json_file:
#     json_data = json.load(json_file)
#     json_file.close()

with open('../data/SnapshotSerengeti_S1-11_v2.1.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

empty_images = list(set([annot['image_id'] for annot in json_data['annotations'] if annot['category_id'] == 0]))

import random
random.shuffle(empty_images)

batchsize = 100
from_ = 0

n = 5000

for i in range(from_, n, batchsize):
    batch = empty_images[i:i+batchsize]
    open("download_file_list.txt", 'w').write(".JPG\n".join(batch))
    os.system("../lib/azcopy/azcopy cp  https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped/ '../data/empty/' --list-of-files download_file_list.txt --output-level quiet")
    print('#remaining empty images', n - i-batchsize, '#next_i', i+batchsize)