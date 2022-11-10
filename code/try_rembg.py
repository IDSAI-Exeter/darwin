from rembg import remove
import cv2



if __name__ == '__main__':
    from imageio.v2 import imread
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    json_data = None
    label_dir = '../data/serengeti_bboxes/labels/'
    image_dir = '../data/serengeti_bboxes/images/'
    data_dir = '../data/'
    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    image_ids = list(set([annot['image_id'] for annot in json_data]))

    image_id = image_ids[15]
    filename = image_id.split('/')[-1]
    bboxes = [annot for annot in json_data if annot['image_id'] == image_id]

    x, y, w, h = bboxes[0]['bbox']  # origin at upper-left

    print(x,y,w,h)

    # We want to change this to use the bbox produced by yolo

    image = cv2.imread(image_dir + filename + '.JPG')

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    crop_img = image[y:y + h, x:x + w]

    output = remove(crop_img)

    cv2.imshow("rembg", output)

    mask = np.zeros(image.shape[:2], dtype="uint8")

    #bbox = np.array([x, y, x+w, y, x+w, y+h, x, y+h])

    bbox = (int(x), int(y), int(x+w), int(y+h))

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

    cv2.imshow('image', image)

    cv2.waitKey(0)