import json
import os

json_data = None

with open('../data/bbox_species.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

bbox_files = list(set([annot['image']['file_name'] for annot in json_data]))

batchsize = 100
from_ = 0

for i in range(from_, len(bbox_files), batchsize):
    batch = bbox_files[i:i+batchsize]
    open("download_file_list.txt", 'w').write("\n".join(batch))
    os.system("../lib/azcopy/azcopy cp  https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped/ '../data/images/' --list-of-files download_file_list.txt --output-level quiet")
    print('#remainingimagesbbox', len(bbox_files) - i-batchsize, '#next_i', i+batchsize)