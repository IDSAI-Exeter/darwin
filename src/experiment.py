# generate train/val/test sets in separate directories and a yaml experiment file
import json
import os
import random
from collections import Counter
import pandas

n = 1

dataset_dir = "../data/serengeti_bboxes/"

discard_file = "../discard.json"

species_bbox_file = '../data/bbox_species.json'


def main(experiment_dir, n_images, selected_species):

    n_images_val = n_images

    try:
        os.mkdir("../data/experiments")
    except:
        pass
    try:
        os.mkdir(experiment_dir)
    except:
        pass
    for d in ['train/', 'val/', 'test/']:
        try:
            os.mkdir(experiment_dir + d)
            os.mkdir(experiment_dir + d + 'images')
            os.mkdir(experiment_dir + d + 'labels')
        except:
            pass
    try:
        os.mkdir(experiment_dir + 'train/runs/')
    except:
        pass

    bbox_data = None
    with open(species_bbox_file) as json_file:
        bbox_data = json.load(json_file)
        json_file.close()

    discard_data = None
    with open(discard_file) as json_file:
        discard_data = json.load(json_file)
        json_file.close()

    species_file = '../data/serengeti_bboxes/species_classes.json'
    species_classes = json.load(open(species_file))

    # get species list
    species = list(set([bbox['species'] for bbox in bbox_data]))
    species.remove('empty')
    species.remove('human')

    # A random 80/10/10 split

    train_set = []
    test_set = []
    val_set = []

    # remove missing images from bbox
    print('bboxes : ', len(bbox_data))
    bbox_data = [bbox for bbox in bbox_data if os.path.isfile(dataset_dir + 'images/' + bbox["image_id"].split('/')[-1] + '.JPG')]
    print('bboxes with file :', len(bbox_data))

    # discard locations with a resizing bbox issue
    bbox_data = [bbox for bbox in bbox_data if {'season': bbox['annotation']['season'], 'location': bbox['annotation']['location']} not in discard_data]

    print('bboxes clean :', len(bbox_data))

    species_counts = []

    # Randomly select locations? 80/10/10

    locations = list(set([bbox['annotation']['location'] for bbox in bbox_data]))

    random.shuffle(locations)

    test_locations = locations[0:int(len(locations)/5)]
    train_locations = [loc for loc in locations if loc not in test_locations]

    for sp in selected_species:
        individuals = [bbox for bbox in bbox_data if bbox['annotation']['location'] in train_locations and bbox['species'] == sp]
        images = [bbox['image_id'] for bbox in individuals]
        counts = Counter(images)

        n_total = len(individuals)

        shuffled = list(set(images))
        random.shuffle(shuffled)

        n = sum(counts.values())
        i = 0

        if n < 10:
            continue

        while i < n_images and shuffled:
            train_set.append(shuffled[0])
            i += 1
            shuffled = shuffled[1:]

        n_train_set = i

        # Generate test set
        individuals = [bbox for bbox in bbox_data if bbox['annotation']['location'] in test_locations
                       and bbox['species'] == sp]
        images = [bbox['image_id'] for bbox in individuals]
        counts = Counter(images)
        shuffled = list(set(images))
        random.shuffle(shuffled)

        n_total += len(individuals)
        i = 0
        while shuffled:
            test_set.append(shuffled[0])
            # i += counts[shuffled[0]]
            i += 1
            shuffled = shuffled[1:]

        n_test_set = i

        # Generate val set
        individuals = [bbox for bbox in bbox_data if bbox['annotation']['location'] in test_locations
                       and bbox['species'] == sp]
        images = [bbox['image_id'] for bbox in individuals]
        counts = Counter(images)
        shuffled = list(set(images))
        random.shuffle(shuffled)
        i = 0

        while i < n_images_val and shuffled: #0.020:
            val_set.append(shuffled[0])
            i += 1
            shuffled = shuffled[1:]

        n_val_set = i

        species_counts.append({'species': sp, 'total': n_total, 'train': n_train_set, 'test': n_test_set, 'val' : n_val_set})

    print('# train:', len(train_set), ' # val:', len(val_set), '# test :', len(test_set))
    species_counts = pandas.DataFrame(species_counts)
    species_counts.to_csv(experiment_dir + 'counts.csv')

    # for image_id in train_set:
    #     filename = image_id.split('/')[-1]
    #     d = 'train/'
    #     #os.system("convert -size 640 %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'+filename+'.JPG'))
    #     os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
    #     os.system("cp %s %s"%(dataset_dir+'species_labels/'+filename+'.txt', experiment_dir+d+'labels/'))

    for image_id in test_set:
        filename = image_id.split('/')[-1]
        d = 'test/'
        os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
        os.system("cp %s %s"%(dataset_dir+'species_labels/'+filename+'.txt', experiment_dir+d+'labels/'))

    for image_id in val_set:
        filename = image_id.split('/')[-1]
        d = 'val/'
        os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
        os.system("cp %s %s"%(dataset_dir+'species_labels/'+filename+'.txt', experiment_dir+d+'labels/'))

    with open(experiment_dir + "test_locations.json", 'w') as json_file:
        json.dump(test_locations, json_file)

    with open(experiment_dir + "train_locations.json", 'w') as json_file:
        json.dump(train_locations, json_file)

    with open(experiment_dir + "train.yaml", 'w') as yaml_file:
        yaml_file.write("path: ../%s\n"%experiment_dir)
        yaml_file.write("train: train/images/\n")
        yaml_file.write("test: test/images/\n")
        yaml_file.write("val: val/images/\n")
        yaml_file.write("\n")
        yaml_file.write("names:\n")
        for k, v in species_classes.items():
            yaml_file.write("   %i: %s\n"%(v,k))
        yaml_file.close()

    print("next download empty images")


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    n_images = 0
    argv = sys.argv[1:]
    species = []

    try:
        opts, args = getopt.getopt(argv, "he:n:s:", ["experiment_dir=", "n_images=", "species="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -n <n_images> -s <species>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -n <n_images> -s <species>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-n", "--n_images"):
            n_images = int(arg)
        elif opt in ("-s", "--species"):
            species = [s.strip().lower() for s in arg.split(',')]

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'
    main(experiment_dir, n_images, species)

