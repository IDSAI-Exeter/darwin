import glob
import os

import cv2
from scipy import ndimage
import random
import json
from rembg import remove
from PIL import Image
import numpy as np
from collections import Counter
import json
config = json.load(open('../config.json'))

dataset_dir = "../data/serengeti_bboxes/"


def flatten(l):
    return [item for sublist in l for item in sublist]


def tighten_to_visible(img):
    pil_image = Image.fromarray(img)
    imageBox = pil_image.getbbox()
    pil_image = pil_image.crop(imageBox)
    nimg = np.array(pil_image)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2RGBA)  # cv::COLOR_RGBA2BGRA
    return ocvim


def main(experiment_dir, empty_imgs_dir, raw_sizes, aug_factors, selected_species):

    fold = experiment_dir.split('/')[-1]

    n_segments = max(raw_sizes)

    train_locations_file = experiment_dir + 'train_locations.json'
    train_locations = None
    with open(train_locations_file) as json_file:
        train_locations = json.load(json_file)
        json_file.close()

    i = 0
    empty_images = glob.glob(empty_imgs_dir + '*.JPG')

    try:
        os.mkdir(augmented_dir)
        os.mkdir(augmented_dir + 'images')
        os.mkdir(augmented_dir + 'labels')
    except:
        pass

    import yaml
    classes = {}

    with open(experiment_dir + 'train.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        classes = {v: k for k, v in data['names'].items()}

    json_data = None
    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    bbox_data = json_data

    species_file = '../data/serengeti_bboxes/species_classes.json'
    species_classes = json.load(open(species_file))

    segments_data = None
    with open('../segments.json') as json_file:
        segments_data = json.load(json_file)
        json_file.close()

    high_quality = [segment['image_id'] for segment in segments_data if segment['quality'] == 'high']

    from collections import defaultdict
    species_segments = defaultdict(list)

    for sp in selected_species:
        individuals = [bbox for bbox in bbox_data if bbox['annotation']['location'] in train_locations and bbox['species'] == sp]
        images = [bbox['image_id'] for bbox in individuals]
        images = [image_id for image_id in images if image_id.split('/')[-1] in high_quality]
        counts = Counter(images)

        n_total = len(individuals)

        shuffled = list(set(images))
        random.shuffle(shuffled)

        n = sum(counts.values())
        i = 0

        while i < n_segments and shuffled:
            image_id = shuffled[0]
            shuffled = shuffled[1:]
            image_id = image_id.split('/')[-1]

            bboxes = [annot for annot in json_data if annot['image_id'].split('/')[-1] == image_id]

            # restrict to images with only one bbox
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

                        crop_img = image[y:y + h, x:x + w]
                        crop_img = image[by:by + bh, bx:bx + bw]
                        animal_image = remove(crop_img) #, alpha_matting=True)

                        animal_image = animal_image[y-by:y-by+h, x-bx:x-bx+w]

                        animal_image = tighten_to_visible(animal_image)

                        mask = animal_image[:, :, 3]
                        alpha_ratio = sum(flatten([[1 if p > 250 else 0 for p in l] for l in mask])) / sum(flatten([[1 for p in l] for l in mask]))

                        # cv2.imshow('augment', image)
                        # cv2.waitKey(0)

                        if alpha_ratio > 0.3 and animal_image.shape[0] * animal_image.shape[1] > 50*50: #100*100:
                            species_segments[sp].append({'segment': animal_image, 'image_id': image_id})
                            i += 1

    # Generate copy pasted training images

    for r in raw_sizes:
        raw_dir = experiment_dir + 'raw_' + str(r) + '/'
        segments = {}
        d = 'raw_%i/' % r

        try:
            os.mkdir(experiment_dir + d)
            os.mkdir(experiment_dir + d + 'images')
            os.mkdir(experiment_dir + d + 'labels')
        except:
            pass

        for k in species_segments.keys():
            random.shuffle(species_segments[k])
            segments[k] = species_segments[k][:r]
            for segment in segments[k]:
                filename = segment['image_id']
                os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
                os.system("cp %s %s"%(dataset_dir+'species_labels/'+filename+'.txt', experiment_dir+d+'labels/'))

        for a in aug_factors:
            augmented_dir = experiment_dir + 'augment_' + str(r) + '_' + str(a) + '/'
            try:
                os.mkdir(augmented_dir)
                os.mkdir(augmented_dir + 'images')
                os.mkdir(augmented_dir + 'labels')
            except:
                pass

            os.system("cp %s %s"%(experiment_dir + d + 'images/*.JPG', augmented_dir + 'images/'))
            os.system("cp %s %s"%(experiment_dir + d + 'labels/*.txt', augmented_dir + 'labels/'))

            for k in species_segments.keys():
                max_augmented = int(r*a / len(segments[k]))
                if not max_augmented:
                    max_augmented = 1

                j = 0
                for segment in segments[k]:
                    animal_image = segment['segment']
                    n = j + max_augmented
                    while j < n:

                        # This would be where to do a proper selection of the empty images based
                        # on time of day of the segment and time of day of the empty image
                        # this can be done as a list comprehension on empty_images and using annotations

                        n_rand = random.randint(0, len(empty_images)-1)
                        empty_image = cv2.imread(empty_images[n_rand])
                        s_img = animal_image.copy()
                        l_img = empty_image

                        x_offset = y_offset = 50

                        scale_percent = random.randint(95, 105)
                        width = int(s_img.shape[1] * scale_percent / 100)
                        height = int(s_img.shape[0] * scale_percent / 100)
                        dim = (width, height)

                        if random.randint(0, 1):
                            s_img = cv2.flip(s_img, 1)

                        s_img = cv2.resize(s_img, dim)

                        angle = random.randint(-5, 5)

                        s_img = ndimage.rotate(s_img, angle)

                        s_img = tighten_to_visible(s_img)

                        if (l_img.shape[1] - s_img.shape[1] > 0) and (l_img.shape[0] - s_img.shape[0] > 0):
                            x_offset = random.randint(0, l_img.shape[1] - s_img.shape[1])
                            y_min = 0
                            y_banner = 100
                            if (l_img.shape[0]/2.0 - s_img.shape[0]) > 0:
                                y_min = l_img.shape[0]/2.0 - s_img.shape[0]

                            y_offset = random.randint(y_min, l_img.shape[0] - s_img.shape[0] - y_banner)

                            y1, y2 = y_offset, y_offset + s_img.shape[0]
                            x1, x2 = x_offset, x_offset + s_img.shape[1]

                            alpha_s = s_img[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            import numpy
                            #alpha_m = numpy.array([[1 if p != 0 else 0 for p in l] for l in s_img[:, :, 3]])
                            #alpha_l = 1.0 - alpha_m

                            for c in range(0, 3):
                                l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])


                            # cv2.rectangle(l_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                            cv2.imwrite(augmented_dir + '/images/' + str(classes[k]) + '_' + str(j) + '.JPG', l_img)

                            # convert to center x, center y, w, h
                            cx = x1 + s_img.shape[1] / 2.0
                            cy = y1 + s_img.shape[0] / 2.0

                            # Normalise
                            cx = cx / l_img.shape[1]
                            w_ = s_img.shape[1] / l_img.shape[1]
                            cy = cy / l_img.shape[0]
                            h_ = s_img.shape[0] / l_img.shape[0]

                            str_ = '%i %f %f %f %f\n' % (classes[k], cx, cy, w_, h_)
                            open(augmented_dir + '/labels/' + str(classes[k]) + '_' + str(j) + '.txt', 'w').write(str_)
                            j += 1

            with open(experiment_dir + "augment_%i_%i.yaml"%(r, a), 'w') as yaml_file:
                yaml_file.write("path: ../%s\n"%experiment_dir)
                yaml_file.write("train: augment_%i_%i/images/\n"%(r, a))
                yaml_file.write("test: test/images/\n")
                yaml_file.write("val: val/images/\n")
                yaml_file.write("\n")
                yaml_file.write("names:\n")
                for k, v in classes.items():
                    yaml_file.write("   %i: %s\n"%(v, k))
                yaml_file.close()

            with open(experiment_dir + "sbatch_augment_%i_%i.sh"%(r, a), 'w') as file:
                file.write("#!/bin/bash\n")
                file.write("#SBATCH --partition=small\n")
                file.write("#SBATCH --nodes=1\n")
                file.write("#SBATCH --gres=gpu:1\n")
                file.write("#SBATCH --mail-type=ALL\n")
                file.write("#SBATCH --mail-user=%s\n"%config['email'])
                file.write("source ../../../../../../.profile\n")
                file.write("source ../../../../darwin_venv/bin/activate\n")
                file.write("echo 'training on %i * %i augmented trainset'\n"%(r, a))
                # file.write("python3 ../../../../lib/yolov5/train.py --epochs 10000 --patience 300 --data augment_%i_%i.yaml --project augment_%i_%i/runs/ --name augment --batch 16\n"%(r, a, r, a))
                file.write("python3 ../../../../lib/yolov5/train.py --epochs 300 --data augment_%i_%i.yaml --project augment_%i_%i/runs/ --name augment --batch 16\n"%(r, a, r, a))
                file.write("\n")
                file.close()

            with open(experiment_dir + "resume_augment_%i_%i.sh"%(r, a), 'w') as file:
                file.write("#!/bin/bash\n")
                file.write("#SBATCH --partition=small\n")
                file.write("#SBATCH --nodes=1\n")
                file.write("#SBATCH --gres=gpu:1\n")
                file.write("#SBATCH --mail-type=ALL\n")
                file.write("#SBATCH --mail-user=%s\n"%config['email'])
                file.write("source ../../../../../../.profile\n")
                file.write("source ../../../../darwin_venv/bin/activate\n")
                file.write("echo 'resume training on %i * %i augmented trainset'\n"%(r, a))
                file.write("python3 ../../../../lib/yolov5/train.py --resume --exist-ok --epochs 300 --data augment_%i_%i.yaml --project augment_%i_%i/runs/ --name augment --batch 16\n"%(r, a, r, a))
                file.write("\n")
                file.close()

            with open(experiment_dir + "validate_augment_%i_%i.sh"%(r, a), 'w') as file:
                file.write("#!/bin/bash\n")
                file.write("#SBATCH --partition=small\n")
                file.write("#SBATCH --nodes=1\n")
                file.write("#SBATCH --gres=gpu:1\n")
                file.write("#SBATCH --mail-type=ALL\n")
                file.write("#SBATCH --mail-user=%s\n"%config['email'])
                file.write("#SBATCH --output=validate-%j.out\n")
                file.write("source ../../../../../../.profile\n")
                file.write("source ../../../../darwin_venv/bin/activate\n")
                file.write("echo 'validate(%i * %i) augmented trainset'\n"%(r, a))
                file.write("python3 ../../../../lib/yolov5/val.py --data augment_%i_%i.yaml --weights augment_%i_%i/runs/augment/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_augment_%i_%i.out\n"%(r, a, r, a, fold, r, a))
                file.write("\n")
                file.close()

        with open(experiment_dir + "raw_%i.yaml"%r, 'w') as yaml_file:
            yaml_file.write("path: ../%s\n" % experiment_dir)
            yaml_file.write("train: raw_%i/images/\n"%r)
            yaml_file.write("test: test/images/\n")
            yaml_file.write("val: val/images/\n")
            yaml_file.write("\n")
            yaml_file.write("names:\n")
            for k, v in classes.items():
                yaml_file.write("   %i: %s\n" % (v, k))
            yaml_file.close()

        with open(experiment_dir + "sbatch_raw_%i.sh"%r, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --partition=small\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --gres=gpu:1\n")
            file.write("#SBATCH --mail-type=ALL\n")
            file.write("#SBATCH --mail-user=%s\n"%config['email'])
            file.write("source ../../../../../../.profile\n")
            file.write("source ../../../../darwin_venv/bin/activate\n")
            file.write("echo 'training on %i raw trainset'\n" % r)
            # file.write("python3 ../../../../lib/yolov5/train.py --epochs 10000 --patience 300 --data raw_%i.yaml --project raw_%i/runs/ --name raw --batch 16\n"%(r, r))
            file.write("python3 ../../../../lib/yolov5/train.py --epochs 300 --data raw_%i.yaml --project raw_%i/runs/ --name raw --batch 16\n"%(r, r))
            file.write("\n")
            file.close()

        with open(experiment_dir + "resume_raw_%i.sh"%r, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --partition=small\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --gres=gpu:1\n")
            file.write("#SBATCH --mail-type=ALL\n")
            file.write("#SBATCH --mail-user=%s\n"%config['email'])
            file.write("source ../../../../../../.profile\n")
            file.write("source ../../../../darwin_venv/bin/activate\n")
            file.write("echo 'resume training on %i raw trainset'\n" % r)
            file.write(
                "python3 ../../../../lib/yolov5/train.py --resume --exist-ok --epochs 300 --data raw_%i.yaml --project raw_%i/runs/ --name raw --batch 16\n"%(r, r))
            file.write("\n")
            file.close()

        with open(experiment_dir + "validate_raw_%i.sh"%r, 'w') as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --partition=small\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --gres=gpu:1\n")
            file.write("#SBATCH --mail-type=ALL\n")
            file.write("#SBATCH --mail-user=%s\n"%config['email'])
            file.write("#SBATCH --output=validate-%j.out\n")
            file.write("source ../../../../../../.profile\n")
            file.write("source ../../../../darwin_venv/bin/activate\n")
            file.write("echo 'validate(%i) raw trainset'\n"%r)
            file.write("python3 ../../../../lib/yolov5/val.py --data raw_%i.yaml --weights raw_%i/runs/raw/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_raw_%i.out\n"%(r, r, fold, r))
            file.write("\n")
            file.close()


if __name__ == '__main__':
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    aug_factors = [1, 2, 4]
    raw_sizes = [1, 5, 10]  # , 500]
    species = []
    try:
        opts, args = getopt.getopt(argv, "he:r:a:s:", ["experiment_dir=", "raw_sizes=", "aug_factors=", "species="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -r <raw_sizes> -a <aug_factors> -s <species>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -r <raw_sizes> -a <aug_factors> -s <species>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-r", "--raw_sizes"):
            raw_sizes = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-a", "--aug_factors"):
            aug_factors = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-s", "--species"):
            species = [s.strip().lower() for s in arg.split(',')]

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    empty_imgs_dir = experiment_dir + 'empty/'
    main(experiment_dir, empty_imgs_dir, raw_sizes, aug_factors, species)
