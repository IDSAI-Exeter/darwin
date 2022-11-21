def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



test_labels_dir = '../data/experiments/sample/test/labels/'
detect_labels_dir = '../lib/yolov5/runs/detect/exp/labels/'

import glob

test_labels_files = glob.glob(test_labels_dir + '*.txt')
detect_labels_files = glob.glob(detect_labels_dir + '*.txt')

test_labels_dic = {}
detect_labels_dic = {}

for file in test_labels_files:
    test_f = open(file).readlines()
    bboxes =[]
    for line in test_f:
        cl, x, y, w, h = line.split()
        bboxes.append([float(cl), float(x), float(y), float(w), float(h), 1.0])
    test_labels_dic[file.split('/')[-1].split('.')[0]] = bboxes

for file in detect_labels_files:
    f = open(file).readlines()
    bboxes = []
    for line in f:
        cl, x, y, w, h, cf = line.split()
        bboxes.append([float(cl), float(x), float(y), float(w), float(h), float(cf)])
    detect_labels_dic[file.split('/')[-1].split('.')[0]] = bboxes

#print(test_labels_dic)
#print(detect_labels_dic)

bboxes_accuracy = []
log_errors = []
areas = []

for k in test_labels_dic.keys():
    t_bboxes = test_labels_dic[k]
    d_bboxes = []
    try:
        d_bboxes = detect_labels_dic[k]
    except:
        pass
    for cl, x, y, w, h, cf in t_bboxes:
        ious = []
        for cl2, x2, y2, w2, h2, cf2 in d_bboxes:
            ious.append(bb_intersection_over_union([x, y, x+w, y+h], [x2, y2, x2+w2, y2+h2]))
        accuracy = 0
        if ious!=[]:
            accuracy = max(ious)
        bboxes_accuracy.append(accuracy)
        areas.append(w*h)
        import math
        log_errors.append(math.log(1-accuracy))

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# plot = plt.scatter(bboxes_accuracy, areas)
# plt.ylabel('bbox area')
# plt.xlabel('iou accuracy')
# plt.show()
#
# plot = plt.scatter(log_errors, areas)
# plt.ylabel('bbox area')
# plt.xlabel('log iou error')
# plt.show()
#
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#
# axs[0].hist(areas, 10)
# axs[0].set_xlabel('bbox area')
# #axs[0].show()
#
# axs[1].hist(bboxes_accuracy, 10)
# axs[1].set_xlabel('iou accuracy')
# plt.show()

plt.hist2d(bboxes_accuracy, areas)
plt.ylabel('areas')
plt.xlabel('iou accuracy')
plt.show()

bins = [[], [], [], [], [], [], [], [], [], []] # [[]] * 10

from statistics import mean

for area, accuracy in zip(areas, bboxes_accuracy):
    #print(int(area*10), area)
    bins[int(area*10)].append(accuracy)

bins_means = []

for bin in bins:
    print(len(bin))
    bins_means.append(mean(bin))

print(bins_means)

n = 0
for m in bins_means:
    plt.bar(n, m, 0.1, edgecolor='black')
    n += 0.1

plt.xlabel('area')
plt.ylabel('avg iou accuracy')
plt.show()