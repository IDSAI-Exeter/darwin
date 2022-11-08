# generate train/val/test sets in separate directories and a yaml experiment file
import json
import os
import random
from collections import Counter
import pandas

n = 1

dataset_dir = "../data/serengeti_bboxes/"
#experiment_dir = "../data/experiments/sample/"
experiment_dir = "../data/experiments/sample_species/"
train_ratio = .01
test_ratio = .005
val_ratio = .005

def main():
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

    json_data = None
    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    # get species list
    species = list(set([bbox['species'] for bbox in json_data]))


    # A random 80/10/10 split

    train_set = []
    test_set = []
    val_set = []

    # remove missing images from bbox
    print('bboxes : ', len(json_data))
    json_data = [bbox for bbox in json_data if os.path.isfile(dataset_dir + 'images/' + bbox["image_id"].split('/')[-1] + '.JPG')]
    print('bboxes with file :', len(json_data))

    species_counts = []

    # select n images across species
    for sp in species:
        individuals = [bbox for bbox in json_data if bbox['species'] == sp]
        images = [bbox['image_id'] for bbox in individuals]
        counts = Counter(images)

        #print('species :', sp, '# :', len(individuals))
        #print('        # images :', len(counts.keys()))

        shuffled = list(set(images))
        random.shuffle(shuffled)

        # take 80% of the list considering # of individual per image
        n = sum(counts.values())
        i = 0

        if n < 10:
            continue

        while i/n < train_ratio: #0.01:
            train_set.append(shuffled[0])
            i += counts[shuffled[0]]
            shuffled = shuffled[1:]

        n_train_set = i

        i = 0

        while i/n < test_ratio: #0.015:
            val_set.append(shuffled[0])
            i += counts[shuffled[0]]
            shuffled = shuffled[1:]

        n_test_set = i

        i = 0

        while i/n < val_ratio: #0.020:
            test_set.append(shuffled[0])
            i += counts[shuffled[0]]
            shuffled = shuffled[1:]

        n_val_set = i

        #print('  # train:', len(train_set), ' # val:', len(val_set), '# test :', len(test_set))
        #print('  #train:', n_train_set, ' #val:', n_val_set, '#test :', n_test_set)
        species_counts.append({'species': sp, 'total': len(individuals), 'train' : n_train_set, 'test' : n_test_set, 'val' : n_val_set})

    print('# train:', len(train_set), ' # val:', len(val_set), '# test :', len(test_set))
    species_counts = pandas.DataFrame(species_counts)
    species_counts.to_csv(experiment_dir + 'counts.csv')

    # for image_id in train_set:
    #     filename = image_id.split('/')[-1]
    #     d = 'train/'
    #     #os.system("convert -size 640 %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'+filename+'.JPG'))
    #     os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
    #     os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels/'))
    #
    # for image_id in test_set:
    #     filename = image_id.split('/')[-1]
    #     d = 'test/'
    #     #os.system("convert -size 640 %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'+filename+'.JPG'))
    #     os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
    #     os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels/'))
    #
    # for image_id in val_set:
    #     filename = image_id.split('/')[-1]
    #     d = 'val/'
    #     #os.system("convert -size 640 %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'+filename+'.JPG'))
    #     os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images/'))
    #     os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels/'))
    #
    # with open(experiment_dir + "experiment.yaml",'w') as yaml_file:
    #     yaml_file.write("path: ../%s\n"%experiment_dir)
    #     yaml_file.write("train: train/images/\n")
    #     yaml_file.write("test: test/images/\n")
    #     yaml_file.write("val: val/images/\n")
    #     yaml_file.write("\n")
    #     yaml_file.write("names:\n")
    #     yaml_file.write("   0: animal\n")
    #     yaml_file.write("   1: vehicule\n")
    #     yaml_file.close()
    #
if __name__ == "__main__":
    main()