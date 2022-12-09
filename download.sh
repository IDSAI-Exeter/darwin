mkdir data
cd data/
wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip
unzip SnapshotSerengeti_S1-11_v2_1.json.zip

wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengetiBboxes_20190903.json.zip
unzip SnapshotSerengetiBboxes_20190903.json.zip

mkdir data/serengeti_bboxes
mkdir data/serengeti_bboxes/images
find data/images/ -type f -exec cp {} data/serengeti_bboxes/images/ \;
