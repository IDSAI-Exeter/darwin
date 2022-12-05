# call example
# python3 generate_augmented_trainset.py -t ../data/experiments/sample_species -e ../data/empty_images/ -a ../data/experiments/sample_species/augmented

import glob
import os

import cv2
from scipy import ndimage
import random
import json
from rembg import remove
from PIL import Image
import numpy as np


def flatten(l):
    return [item for sublist in l for item in sublist]

def main(experiment_dir, empty_imgs_dir, augmented_dir, n_augment):

    print(experiment_dir, empty_imgs_dir)
    i = 0
    #training_set = os.listdir(train_dir)

    train_files = glob.glob(experiment_dir + 'train' + '/images/*.JPG')
    train_labels = glob.glob(experiment_dir + 'train' + '/labels/*.txt')
    empty_images = glob.glob(empty_imgs_dir + '*.JPG')

    try:
        os.mkdir(augmented_dir)
        os.mkdir(augmented_dir + 'images')
        os.mkdir(augmented_dir + 'labels')
    except:
        pass

    # cp -r train augmented
    os.system("cp %s %s"%(experiment_dir + '/train' + '/images/*.JPG', augmented_dir + '/images/'))
    os.system("cp %s %s"%(experiment_dir + '/train' + '/labels/*.txt', augmented_dir + '/labels/'))

    # animal_dir = '../data/animals/'
    # animal_image = cv2.imread(animals[15])
    # animals = glob.glob(animal_dir + '*.jpg')

    import yaml
    classes = {}

    with open(experiment_dir + 'train.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        classes = {v: k for k, v in data['names'].items()}

    json_data = None

    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    species_file = '../data/serengeti_bboxes/species_classes.json'
    species_classes = json.load(open(species_file))
    #species_classes = classes

    from collections import defaultdict
    species_segments = defaultdict(list)

    random.shuffle(train_files)

    for train_file in train_files: #[0:100]:
        image_id = train_file.split('/')[-1].split('.')[0]

        #print(train_file, image_id)
        #filename = image_id.split('/')[-1]

        bboxes = [annot for annot in json_data if annot['image_id'].split('/')[-1] == image_id]

        #for bbox in bboxes:
        # restrict to images with only one bbod
        if len(bboxes) == 1:
            bbox = bboxes[0]
            x, y, w, h = bbox['bbox']

            r = 0.05

            bx, by, bw, bh = x - w*r, y - h*r, w + w*2*r, h + h*2*r
            #if x > 50 and y > 50 and x+w < bbox['image']['width'] - 50 and y+h < bbox['image']['height'] - 50:
            #if bx > 50 and by > 50 and bx + bw < bbox['image']['width'] - 50 and by + bh < bbox['image']['height'] - 50:
            if bx > 0 and by > 0 and bx + bw < bbox['image']['width'] and by + bh < bbox['image']['height'] - 200:
                image = cv2.imread(train_file)
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

                mask = animal_image[:, :, 3]
                alpha_ratio = sum(flatten([[1 if p > 200 else 0 for p in l] for l in mask])) / sum(flatten([[1 for p in l] for l in mask]))

                # cv2.imshow('augment', image)
                # cv2.waitKey(0)

                if alpha_ratio > 0.3:

                    species_segments[bbox['species']].append({'segment': animal_image})

    # after browsing training files
    # read from input now n_augment = 1000

    for k in species_segments.keys():
        n_segments = len(species_segments[k])

        max_augmented = int(n_augment / n_segments)
        if not max_augmented:
            max_augmented = 1
        print(k, n_segments, n_augment, max_augmented)
        i += 1
        j = 0
        for segment in species_segments[k]:
            animal_image = segment['segment']
            n = j + max_augmented
            #for n in range(0, max_augmented): #random.randint(0, max_augmented)):
            while j < n:

                #print(i, j)
                n_rand = random.randint(0, len(empty_images)-1)
                empty_image = cv2.imread(empty_images[n_rand])
                s_img = animal_image.copy()
                l_img = empty_image

                x_offset = y_offset = 50

                scale_percent = random.randint(80, 200)
                width = int(s_img.shape[1] * scale_percent / 100)
                height = int(s_img.shape[0] * scale_percent / 100)
                dim = (width, height)

                if random.randint(0, 1):
                    s_img = cv2.flip(s_img, 1)

                s_img = cv2.resize(s_img, dim)

                angle = random.randint(-30, 30)

                s_img = ndimage.rotate(s_img, angle)

                pil_image = Image.fromarray(s_img)
                imageBox = pil_image.getbbox()
                pil_image = pil_image.crop(imageBox)
                nimg = np.array(pil_image)
                ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGBA2BGRA)  # cv::COLOR_RGBA2BGRA
                s_img = ocvim

                if (l_img.shape[1] - s_img.shape[1] > 0) and (l_img.shape[0] - s_img.shape[0] > 0 ):
                    x_offset = random.randint(0, l_img.shape[1] - s_img.shape[1])
                    y_min = 0
                    if (l_img.shape[0]/2.0 - s_img.shape[0]) > 0:
                        y_min = l_img.shape[0]/2.0 - s_img.shape[0]

                    y_offset = random.randint(y_min, l_img.shape[0] - s_img.shape[0])

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

                    cv2.imwrite(augmented_dir + '/images/' + str(i) + '_' + str(j) + '.JPG', l_img)

                    # convert to center x, center y, w, h
                    cx = x1 + s_img.shape[1] / 2.0
                    cy = y1 + s_img.shape[0] / 2.0

                    # Normalise
                    cx = cx / l_img.shape[1]
                    w_ = s_img.shape[1] / l_img.shape[1]
                    cy = cy / l_img.shape[0]
                    h_ = s_img.shape[0] / l_img.shape[0]

                    str_ = '%i %f %f %f %f\n' % (classes[k], cx, cy, w_, h_)
                    open(augmented_dir + '/labels/' + str(i) + '_' + str(j) + '.txt', 'w').write(str_)
                    # cv2.imshow('augment', l_img)
                    # cv2.waitKey(0)
                    j += 1

    with open(experiment_dir + "augment_%i.yaml"%n_augment, 'w') as yaml_file:
        yaml_file.write("path: ../%s\n"%experiment_dir)
        yaml_file.write("train: augment_%i/images/\n"%n_augment)
        yaml_file.write("test: test/images/\n")
        yaml_file.write("val: test/images/\n")
        yaml_file.write("\n")
        yaml_file.write("names:\n")
        for k, v in species_classes.items():
            yaml_file.write("   %i: %s\n"%(v,k))
        yaml_file.close()

    with open(experiment_dir + "sbatch_augment_%i.sh"%n_augment, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --partition=small\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --gres=gpu:1\n")
        file.write("#SBATCH --mail-type=ALL\n")
        file.write("#SBATCH --mail-user=cedric.mesnage@gmail.com\n")
        file.write("source ../../../../../.profile\n")
        file.write("source ../../../darwin_venv/bin/activate\n")
        file.write("echo 'training on %i augmented trainset'\n"%n_augment)
        file.write("python3 ../../../lib/yolov5/train.py --data augment_%i.yaml --project augment_%i/runs/ --name augment --batch 16\n"%(n_augment, n_augment))
        file.write("\n")
        file.close()

if __name__ == '__main__':
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    n_augment = 0

    try:
        opts, args = getopt.getopt(argv, "he:i:", ["experiment_dir=", "n_images"])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -i <n_images>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -i <n_images>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-i", "--n_images"):
            n_augment = int(arg)
    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    empty_imgs_dir = experiment_dir + 'empty/'
    augmented_dir = experiment_dir + 'augment_' + str(n_augment) + '/'
    main(experiment_dir, empty_imgs_dir, augmented_dir, n_augment)
