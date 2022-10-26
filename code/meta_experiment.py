# generate train/val/test sets in separate directories and a yaml experiment file
import json
import os
import random

n = 1

dataset_dir = "../data/serengeti_bboxes/"

experiment_dir = "../data/experiments/n1/"

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

    train_set = []
    test_set = []
    val_set = []

    # remove missing images from bbox
    print('bboxes : ', len(json_data))
    json_data = [bbox for bbox in json_data if os.path.isfile(dataset_dir + 'images/' + bbox["image_id"].split('/')[-1] + '.JPG')]
    print('bboxes with file :', len(json_data))

    # select n images across species
    for sp in species:
        individuals = [bbox for bbox in json_data if bbox['species'] == sp]
        print('species : ', sp, '# :', len(individuals))
        if len(individuals) > n*3:
            sample = random.sample(individuals, n*3)
            train_set += sample[0:int(len(sample)/3)]
            val_set += sample[int(len(sample)/3):2*int(len(sample)/3)]
            test_set += sample[2*int(len(sample)/3):]

    for bbox in train_set:
        filename = bbox["image_id"].split('/')[-1]
        d = 'train/'
        os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images'))
        os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels'))

    for bbox in test_set:
        filename = bbox["image_id"].split('/')[-1]
        d = 'test/'
        os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images'))
        os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels'))

    for bbox in val_set:
        filename = bbox["image_id"].split('/')[-1]
        d = 'val/'
        os.system("cp %s %s"%(dataset_dir+'images/'+filename+'.JPG', experiment_dir+d+'images'))
        os.system("cp %s %s"%(dataset_dir+'labels/'+filename+'.txt', experiment_dir+d+'labels'))

    with open(experiment_dir + "experiment.yaml",'w') as yaml_file:
        yaml_file.write("path: ../%s\n"%experiment_dir)
        yaml_file.write("train: train/images/\n")
        yaml_file.write("test: test/images/\n")
        yaml_file.write("val: val/images/\n")
        yaml_file.write("\n")
        yaml_file.write("names:\n")
        yaml_file.write("   0: animal\n")
        yaml_file.write("   1: vehicule\n")
        yaml_file.close()

if __name__ == "__main__":
    main()