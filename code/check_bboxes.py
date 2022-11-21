import json
import cv2


image_id = 'S5_G07_R2_IMAG0454'

json_data = None
with open('../data/bbox_species.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

#bboxes = [annot for annot in json_data if annot['image_id'].split('/')[-1] == image_id]


bboxes = [annot for annot in json_data if annot['annotation']['season'] == 'S6' and annot['annotation']['location'] == 'G07']

print(bboxes)

#image = cv2.imread('../data/serengeti_bboxes/images/'+image_id + '.JPG')


for bbox in bboxes:
    x, y, w, h = bbox['bbox']

    image = cv2.imread('../data/serengeti_bboxes/images/'+bbox['image_id'].split('/')[-1] + '.JPG')
    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 3)

    cv2.imshow('image', image)
    cv2.waitKey(0)