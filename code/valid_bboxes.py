import json
import cv2

json_data = None
with open('../data/bbox_species.json') as json_file:
    json_data = json.load(json_file)
    json_file.close()

season = 'S5' #S06
location = 'C11' #'G07'

seasons = sorted(list(set([annot['annotation']['season'] for annot in json_data])))

discard = []
keep = []

for season in seasons:
    data = [annot for annot in json_data if annot['annotation']['season'] == season ]
    locations = list(set([annot['annotation']['location'] for annot in data]))

    for location in locations:
        print(season, location)
        location_bboxes = [annot for annot in data if annot['annotation']['location'] == location]
        #image = cv2.imread('../data/serengeti_bboxes/images/'+image_id + '.JPG')
        images = list(set([bbox['image_id'] for bbox in location_bboxes]))

        validated = False
        while not validated:

            for image_id in images:

                image = cv2.imread('../data/serengeti_bboxes/images/'+image_id.split('/')[-1] + '.JPG')
                bboxes = [bbox for bbox in location_bboxes if bbox['image_id'] == image_id]

                for bbox in bboxes:
                    x, y, w, h = bbox['bbox']
                    #image = cv2.imread('../data/serengeti_bboxes/images/'+bbox['image_id'].split('/')[-1] + '.JPG')
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 3)
                try:
                    cv2.imshow('image', image)
                except:
                    continue
                    print('file error')

                key = cv2.waitKey(0)
                if key == 32:
                    continue
                elif key == 107:
                    # keep
                    keep.append({'season': season, 'location': location})
                    validated = True
                    break
                elif key == 100:
                    # discard
                    discard.append({'season': season, 'location': location})
                    validated = True
                    break

        json.dump(discard, open('discard.json', 'w'))
        json.dump(keep, open('keep.json', 'w'))
