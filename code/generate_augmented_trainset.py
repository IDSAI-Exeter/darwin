import glob
import os

import cv2
from scipy import ndimage
import random
import json
from rembg import remove

def flatten(l):
    return [item for sublist in l for item in sublist]

def main(train_dir, empty_imgs_dir, augmented_dir):

    print(train_dir, empty_imgs_dir)
    i = 0
    #training_set = os.listdir(train_dir)

    train_files = glob.glob(train_dir + '/train' + '/images/*.JPG')
    train_labels = glob.glob(train_dir + '/train' + '/labels/*.txt')
    empty_images = glob.glob(empty_imgs_dir + '*.JPG')

    try:
        os.mkdir(augmented_dir)
        os.mkdir(augmented_dir + '/images')
        os.mkdir(augmented_dir + '/labels')
    except:
        pass

    # cp -r train augmented

    # animal_dir = '../data/animals/'
    # animal_image = cv2.imread(animals[15])
    # animals = glob.glob(animal_dir + '*.jpg')

    import yaml
    classes = {}

    with open(train_dir + '/experiment.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        classes = {v: k for k, v in data['names'].items()}

    json_data = None

    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    for train_file in train_files:
        image_id = train_file.split('/')[-1].split('.')[0]
        print(train_file, image_id)

        #filename = image_id.split('/')[-1]

        bboxes = [annot for annot in json_data if annot['image_id'].split('/')[-1] == image_id]


        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            if x > 50 and y > 50 and x+w < bbox['image']['width'] - 50 and y+h < bbox['image']['height'] - 50:
                image = cv2.imread(train_file)

                #cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 3)

                x = int(x) - 10
                y = int(y) - 10
                w = int(w) + 20
                h = int(h) + 20

                # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 3)

                # cv2.imshow('augment', image)
                # cv2.waitKey(0)

                crop_img = image[y:y + h, x:x + w]
                animal_image = remove(crop_img) #, alpha_matting=True)

                mask = animal_image[:, :, 3]
                alpha_ratio = sum(flatten([[1 if p > 200 else 0 for p in l] for l in mask])) / sum(flatten([[1 for p in l] for l in mask]))

                # cv2.imshow('augment', image)
                # cv2.waitKey(0)

                if alpha_ratio > 0.3:

                    i += 1

                    max_augmented = 10

                    for n in range(0, random.randint(0, max_augmented)):

                        n_rand = random.randint(0, len(empty_images))
                        empty_image = cv2.imread(empty_images[n_rand])
                        s_img = animal_image.copy()
                        l_img = empty_image

                        x_offset = y_offset = 50

                        scale_percent = random.randint(80, 200)
                        width = int(s_img.shape[1] * scale_percent / 100)
                        height = int(s_img.shape[0] * scale_percent / 100)
                        dim = (width, height)

                        s_img = cv2.flip(s_img, random.randint(0, 1))

                        s_img = cv2.resize(s_img, dim)

                        angle = random.randint(-30, 30)

                        s_img = ndimage.rotate(s_img, angle)

                        if (l_img.shape[1] - s_img.shape[1] > 0) and (l_img.shape[0] - s_img.shape[0] > 0 ):
                            x_offset = random.randint(0, l_img.shape[1] - s_img.shape[1])
                            y_offset = random.randint(0, l_img.shape[0] - s_img.shape[0])

                            y1, y2 = y_offset, y_offset + s_img.shape[0]
                            x1, x2 = x_offset, x_offset + s_img.shape[1]

                            alpha_s = s_img[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            import numpy
                            #alpha_m = numpy.array([[1 if p != 0 else 0 for p in l] for l in s_img[:, :, 3]])
                            #alpha_l = 1.0 - alpha_m

                            for c in range(0, 3):
                                #l_img[y1:y2, x1:x2, c] = (alpha_m * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])
                                l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])
                                # l_img[y1:y2, x1:x2, c] = ( s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])


                            #cv2.rectangle(l_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                            cv2.imwrite(augmented_dir + '/images/' + str(i) + '_' + str(n) + '.JPG', l_img)

                            # convert to center x, center y, w, h
                            cx = x1 + s_img.shape[1] / 2.0
                            cy = y1 + s_img.shape[0] / 2.0

                            # Normalise
                            cx = cx / l_img.shape[1]
                            w_ = s_img.shape[1] / l_img.shape[1]
                            cy = cy / l_img.shape[0]
                            h_ = s_img.shape[0] / l_img.shape[0]

                            str_ = '%i %f %f %f %f\n' % (classes[bbox['species']], cx, cy, w_, h_)
                            open(augmented_dir + '/labels/' + str(i) + '_' + str(n) + '.txt', 'w').write(str_)

                            # cv2.imshow('augment', l_img)
                            # cv2.waitKey(0)

if __name__ == '__main__':
    import sys, getopt

    train_dir = ''
    empty_imgs_dir = ''
    augmented_dir = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "ht:e:a:", ["traindir=", "emptyimgsdir=", "augmenteddir"])
    except getopt.GetoptError:
        print('test.py -t <traindir> -e <emptyimgsdir> -a <augmenteddir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <traindir> -e <emptyimgsdir> -a <augmenteddir>')
            sys.exit()
        elif opt in ("-t", "--traindir"):
            train_dir = arg
        elif opt in ("-e", "--emptyimgsdir"):
            empty_imgs_dir = arg
        elif opt in ("-a", "--augmenteddir"):
            augmented_dir = arg

    main(train_dir, empty_imgs_dir, augmented_dir)