from rembg import remove
import cv2
import numpy as np
import json

def remove_bg_from_bbox(image, bbox):
    # Removes the background from the content of the bounding box given in the image given
    # bbox of the form [x, y, w, h]
    # Returns an image of the size of bbox
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])
    crop_img = image[y:y + h, x:x + w]
    output = remove(crop_img)#, alpha_matting=True)#, only_mask=True)
    return output

if __name__ == '__main__':
    from imageio.v2 import imread
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    json_data = None
    label_dir = '../data/serengeti_bboxes/labels/'
    image_dir = '../data/serengeti_bboxes/images/'
    segments_dir = '../data/animals/'
    data_dir = '../data/'

    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    annotations = json_data

    for instance in annotations:
        filename = instance['image_id'].split('/')[-1]
        species = instance['species']
        bbox = instance['bbox']  # origin at upper-left
        image = cv2.imread(image_dir + filename + '.JPG')
        segment = remove_bg_from_bbox(image, bbox)
        cv2.imwrite(segments_dir + species + '_' + filename + '_' + str(int(bbox[0])) + '.jpg', segment)