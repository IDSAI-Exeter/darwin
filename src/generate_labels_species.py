import json
import os


def main():

    label_dir = '../data/serengeti_bboxes/species_labels/'
    data_dir = '../data/'

    try:
        os.mkdir(label_dir)
    except:
        pass

    json_data = None

    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    # get image ids

    image_ids = list(set([annot['image_id'] for annot in json_data]))

    # classes

    species = list(set([bbox['species'] for bbox in json_data]))

    classes = {}

    i = 0

    for species_ in species:
        classes[species_] = i
        i += 1

    species_file = '../data/serengeti_bboxes/species_classes.json'

    json.dump(classes, open(species_file, 'w'))
    # get the bounding boxes, transform them into yolo representation cx,cy,w,h

    for image in image_ids:
        filename = image.split('/')[-1]
        file = open(label_dir+filename+'.txt', 'w')
        bboxes = [annot for annot in json_data if annot['image_id'] == image]
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']  # origin at upper-left
            width = bbox['image']['width']
            height = bbox['image']['height']

            # convert to center x, center y, w, h
            cx = x + w/2.0
            cy = y + h/2.0

            # Normalise
            cx = cx / width
            w = w / width
            cy = cy / height
            h = h / height

            str_ = '%i %f %f %f %f\n'%(classes[bbox['species']], cx, cy, w, h)

            file.write(str_)
        file.close()


if __name__ == '__main__':
    main()
