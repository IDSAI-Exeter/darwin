import glob
import os
import base64

import cv2
from scipy import ndimage
import random
import json
from rembg import remove
from PIL import Image
import numpy as np
from collections import Counter

dataset_dir = "../data/serengeti_bboxes/"

discard_file = "../data/discard.json"

species_bbox_file = '../data/bbox_species.json'


def flatten(l):
    return [item for sublist in l for item in sublist]


def tighten_to_visible(img):
    pil_image = Image.fromarray(img)
    imageBox = pil_image.getbbox()
    pil_image = pil_image.crop(imageBox)
    nimg = np.array(pil_image)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2RGBA)  # cv::COLOR_RGBA2BGRA
    return ocvim


def main():
    segments = []

    bbox_data = None
    with open(species_bbox_file) as json_file:
        bbox_data = json.load(json_file)
        json_file.close()

    discard_data = None
    with open(discard_file) as json_file:
        discard_data = json.load(json_file)
        json_file.close()

    bbox_data = [bbox for bbox in bbox_data if os.path.isfile(dataset_dir + 'images/' + bbox["image_id"].split('/')[-1] + '.JPG')]
    bbox_data = [bbox for bbox in bbox_data if {'season': bbox['annotation']['season'], 'location': bbox['annotation']['location']} not in discard_data]

    images = list(set([bbox['image_id'] for bbox in bbox_data]))

    print("n images: ", len(images))
    shuffled = list(set(images))
    random.shuffle(shuffled)
    i = 0
    n = len(images)

    while shuffled:
        image_id = shuffled[0]
        shuffled = shuffled[1:]
        image_id = image_id.split('/')[-1]

        bboxes = [annot for annot in bbox_data if annot['image_id'].split('/')[-1] == image_id]

        if len(bboxes) == 1:
            bbox = bboxes[0]
            x, y, w, h = bbox['bbox']

            r = 0.05

            bx, by, bw, bh = x - w*r, y - h*r, w + w*2*r, h + h*2*r
            if bx > 0 and by > 0 and bx + bw < bbox['image']['width'] and by + bh < bbox['image']['height'] - 200:
                train_file = dataset_dir+'images/'+image_id+'.JPG'
                image = None
                try:
                    image = cv2.imread(train_file)
                except:
                    pass

                if image is not None:
                    #cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 3)

                    bx = int(bx)
                    by = int(by)
                    bw = int(bw)
                    bh = int(bh)

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    # x = int(x) - 10
                    # y = int(y) - 10
                    # w = int(w) + 20
                    # h = int(h) + 20

                    # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 3)

                    # cv2.imshow('augment', image)
                    # cv2.waitKey(0)

                    crop_img = image[y:y + h, x:x + w]
                    crop_img = image[by:by + bh, bx:bx + bw]
                    animal_image = remove(crop_img) #, alpha_matting=True)

                    animal_image = animal_image[y-by:y-by+h, x-bx:x-bx+w]

                    animal_image = tighten_to_visible(animal_image)

                    mask = animal_image[:, :, 3]
                    alpha_ratio = sum(flatten([[1 if p > 250 else 0 for p in l] for l in mask])) / sum(flatten([[1 for p in l] for l in mask]))

                    if alpha_ratio > 0.3 and animal_image.shape[0] * animal_image.shape[1] > 50*50: #100*100:
                        print(i, '/', n, image_id)
                        cv2.imwrite("../data/segments/%s.JPG"%image_id, animal_image)
                        segments.append({'segment': "../data/segments/%s.JPG"%image_id, 'image_id': image_id, 'quality': 'unknown'})
        i += 1
    json.dump(segments, open('../data/segments.json', 'w'), ensure_ascii=False)


def tinder():
    segments = json.load(open('../data/segments.json'))
    n = len(segments)
    i = 0
    # for segment in segments:
    while i < n:
        print("%i/%i"%(i, n))
        if segments[i]['quality'] != 'unknown':
            i+=1
            continue
        cv2.imshow('segment', cv2.imread(segments[i]['segment']))
        key = cv2.waitKey(0)
        # print(key)
        if key == 104:
            # high
            segments[i]['quality'] = 'high'
        elif key == 108:
            # low
            segments[i]['quality'] = 'low'
        elif key == 112:
            # poor
            segments[i]['quality'] = 'poor'
        elif key == 122:
            # undo
            i -= 2
        i += 1

    json.dump(segments, open('../data/segments.json', 'w'))


if __name__ == '__main__':
    # main()
    tinder()