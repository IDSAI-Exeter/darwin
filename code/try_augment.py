from rembg import remove
import cv2

def flatten(l):
    return [item for sublist in l for item in sublist]

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

    chosen = False
    image_id = None
    filename = None
    bboxes = None
    x, y, w, h = [None, None, None, None]

    while chosen == False:
        import random
        ind = random.randint(0,len(image_ids))
        image_id = image_ids[ind]
        filename = image_id.split('/')[-1]
        bboxes = [annot for annot in json_data if annot['image_id'] == image_id]
        x, y, w, h = bboxes[0]['bbox']  # origin at upper-left
        print(x,y,w,h)
        if x > 50 and y > 50 and x+w < bboxes[0]['image']['width'] - 50 and y+h < bboxes[0]['image']['height'] - 50:
            chosen = True

            image = cv2.imread(image_dir + filename + '.JPG')
            x = int(x) - 10
            y = int(y) - 10
            w = int(w) + 10
            h = int(h) + 10
            crop_img = image[y:y + h, x:x + w]
            output = remove(crop_img)
            mask = output[:, :, 3]
            print(sum(flatten([[1 if a > 255/2.0 else 0 for a in l] for l in mask])), sum(flatten([[1 for p in l] for l in mask])))
            alpha_ratio = sum(flatten([[1 if p != 0 else 0 for p in l] for l in mask])) / sum(flatten([[1 for p in l] for l in mask]))
            print(alpha_ratio)
            if alpha_ratio < 0.7:
                chosen = False

    image = cv2.imread(image_dir + filename + '.JPG')

    x = int(x) - 10
    y = int(y) - 10
    w = int(w) + 10
    h = int(h) + 10

    crop_img = image[y:y + h, x:x + w]

    output = remove(crop_img)

    #cv2.imshow("rembg", output)

    mask = np.zeros(image.shape[:2], dtype="uint8")

    #bbox = np.array([x, y, x+w, y, x+w, y+h, x, y+h])

    bbox = (int(x), int(y), int(x+w), int(y+h))

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

    #cv2.imshow('image', image)

    import cv2
    from scipy import ndimage

    #s_img = cv2.imread("smaller_image.png")
    #l_img = cv2.imread("larger_image.jpg")

    s_img = output
    l_img = image
    #print(s_img)

    x_offset = y_offset = 50
    #l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
    #s_img = cv2.imread("smaller_image.png", -1)

    scale_percent = random.randint(20, 300)
    width = int(s_img.shape[1] * scale_percent / 100)
    height = int(s_img.shape[0] * scale_percent / 100)
    dim = (width, height)

    s_img = cv2.flip(s_img, 1)
    s_img = cv2.resize(s_img, dim)

    angle = random.randint(-45, 45)

    s_img = ndimage.rotate(s_img, angle)

    x_offset = random.randint(0, l_img.shape[1] - s_img.shape[1])
    y_offset = random.randint(0, l_img.shape[0] - s_img.shape[0])

    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])

    cv2.rectangle(l_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow('augment', l_img)

    #cv2.imshow('mask', s_img)

    cv2.waitKey(0)
