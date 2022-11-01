import sys
sys.path.append("../lib/alpha-matting/")
from alpha_matting_segmentation import perform_alpha_matting
#from ocsvm_segmentation import perform_ocsvm_segmentation
from sbbm_segmentation import perform_SBBM_segmentation

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

    image = image_ids[15]
    filename = image.split('/')[-1]
    bboxes = [annot for annot in json_data if annot['image_id'] == image]
    #for bbox in bboxes:
    x, y, w, h = bboxes[0]['bbox']  # origin at upper-left
    print(x,y,w,h)

    # We want to change this to use the bbox produced by yolo

    image = imread(image_dir + filename + '.JPG')
    bbox = np.array([x, y, x+w, y, x+w, y+h, x, y+h])
    print(bbox)

    # from scipy.ndimage import imread
    from imageio.v2 import imread
    import matplotlib.pyplot as plt

    # apply segmentation bag image (from VOT2016 data set)
    #image = imread('images/bag_00000001.jpg')
    #bbox = np.array([334.02, 128.36, 438.19, 188.78,
    #                 396.39, 260.83, 292.23, 200.41])

    mask, [x0, y0, x1, y1] = perform_alpha_matting(image, bbox, return_crop_region=True)

    #mask, [x0, y0, x1, y1] = perform_ocsvm_segmentation(image, bbox, return_crop_region=True)

    #mask, [x0, y0, x1, y1] = perform_SBBM_segmentation(image, bbox, return_crop_region=True)

    # remove pixels from image that are labelled as background
    image_masked = image.copy()
    for d in range(3):
        image_masked[..., d].flat[~mask.ravel()] = 255

    # display original image, segmentation, and segmented image
    images = [image[y0:y1, x0:x1, :], mask[y0:y1, x0:x1], image_masked[y0:y1, x0:x1]]
    titles = ['Original image (cropped)', 'Segmentation', 'Segmented image']

    bbox_pts = np.concatenate((np.reshape(bbox, (-1, 2)), bbox[:2][np.newaxis, :]))
    bbox_pts -= [x0, y0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for image, title, a in zip(images, titles, ax.flat):
        a.imshow(image)
        for i in range(4):
            a.plot(bbox_pts[i:i + 2, 0], bbox_pts[i:i + 2, 1], 'c-', lw=2)
        a.set_title(title)
        a.axis('off')
    plt.show()
