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


#test_labels_dir = '../data/experiments/sample/test/labels/'
#detect_labels_dir = '../lib/yolov5/runs/detect/exp/labels/'


def main(test_labels_dir, detect_labels_dir, plotfilepath):
    import glob

    from statistics import mean

    test_labels_files = glob.glob(test_labels_dir + '*.txt')
    detect_labels_files = glob.glob(detect_labels_dir + '*.txt')

    test_labels_dic = {}
    detect_labels_dic = {}

    for file in test_labels_files:
        test_f = open(file).readlines()
        bboxes = []
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

    import json
    json_data = None
    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    image_species = {}

    for bbox in json_data:
        image_species[bbox['image_id'].split('/')[-1]] = bbox['species']

    bboxes_accuracy = []
    log_errors = []
    areas = []

    species_accuracy = {}

    from collections import Counter
    species_false_negative = Counter()
    species_predictions = Counter()

    for k in test_labels_dic.keys():
        t_bboxes = test_labels_dic[k]
        d_bboxes = []
        try:
            d_bboxes = detect_labels_dic[k]
        except:
            pass

        species_predictions[image_species[k]] += len(d_bboxes)
        print(species_predictions[image_species[k]])

        for cl, x, y, w, h, cf in t_bboxes:
            ious = []
            fn_iou = 0.0
            for cl2, x2, y2, w2, h2, cf2 in d_bboxes:
                iou = bb_intersection_over_union([x, y, x+w, y+h], [x2, y2, x2+w2, y2+h2])
                if iou != 0:
                    fn_iou = iou
                if cl == cl2:
                    ious.append(iou)
                else:
                    ious.append(0.0)
            accuracy = 0

            if fn_iou != 0.0:
                species_false_negative[image_species[k]] += 1

            if ious != []:
                accuracy = max(ious)

            bboxes_accuracy.append(accuracy)
            areas.append(w*h)
            import math
            log_errors.append(math.log(1-accuracy))
            if image_species[k] not in species_accuracy.keys():
                species_accuracy[image_species[k]] = []
            species_accuracy[image_species[k]].append(accuracy)
            print(image_species[k])


    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8})

    x = []
    y = []
    z = []

    for k in species_accuracy.keys():
        #y.append(mean(species_accuracy[k]))
        print(k, species_accuracy[k], species_predictions[k], species_false_negative[k])
        try:
            iou_accuracy = sum(species_accuracy[k])/(species_predictions[k]+species_false_negative[k])
            x.append(k)
            y.append(iou_accuracy)
            z.append(len(species_accuracy[k]))
        except:
            pass

    x = [e for _, e in sorted(zip(y, x))]
    y = sorted(y)

    plt.barh(x, width=y, label=x)
    plt.xlim([0.0, 1.0])
    plt.xlabel('avg iou accuracy')
    plt.tight_layout()
    plt.savefig(plotfilepath)
    with open(plotfilepath.split('.png')[0] + '.json', 'w') as fp:
        json.dump({'y': y, 'x': x}, fp)
        print({'y': y, 'x': x})


if __name__ == '__main__':
    import sys, getopt

    test_labels_dir = ''
    detect_labels_dir = ''
    plotfilepath = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "ht:d:o:", ["testdir=", "detectdir=", "ofile="])
    except getopt.GetoptError:
        print('test.py -t <testlabeldir> -d <detectedtestlabeldir> -o <plotfilename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <testlabeldir> -d <detectedtestlabeldir> -o <plotfilename>')
            sys.exit()
        elif opt in ("-t", "--testdir"):
            test_labels_dir = arg
        elif opt in ("-d", "--detectdir"):
            detect_labels_dir = arg
        elif opt in ("-o", "--ofile"):
            plotfilepath = arg

    main(test_labels_dir, detect_labels_dir, plotfilepath)


    # exit()

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

    # plt.hist2d(bboxes_accuracy, areas)
    # plt.ylabel('areas')
    # plt.xlabel('iou accuracy')
    # plt.show()
    #
    # bins = [[], [], [], [], [], [], [], [], [], []] # [[]] * 10
    #
    # from statistics import mean
    #
    # for area, accuracy in zip(areas, bboxes_accuracy):
    #     #print(int(area*10), area)
    #     bins[int(area*10)].append(accuracy)
    #
    # bins_means = []
    #
    # for bin in bins:
    #     print(len(bin))
    #     bins_means.append(mean(bin))
    #
    # print(bins_means)
    #
    # n = 0
    # for m in bins_means:
    #     plt.bar(n, m, 0.1, edgecolor='black')
    #     n+=0.1
    #
    # plt.xlabel('area')
    # plt.ylabel('avg iou accuracy')
    # plt.show()