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


def main(test_labels_dir, detect_labels_dir, discardfilepath):
    import glob

    from statistics import mean

    discarded = []

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
                if cl == cl2:
                    ious.append(bb_intersection_over_union([x, y, x+w, y+h], [x2, y2, x2+w2, y2+h2]))
                else:
                    ious.append(0.0)

            accuracy = 0

            if ious != []:
                accuracy = max(ious)

            if accuracy < 0.8:
                discarded.append(k)

    json.dump(open(discardfilepath, 'w'), list(set(discarded)))

if __name__ == '__main__':
    import sys, getopt

    test_labels_dir = ''
    detect_labels_dir = ''
    discardfilepath = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "ht:d:o:", ["testdir=", "detectdir=", "ofile="])
    except getopt.GetoptError:
        print('test.py -t <testlabeldir> -d <detectedtestlabeldir> -o <discardfilepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <testlabeldir> -d <detectedtestlabeldir> -o <discardfilepath>')
            sys.exit()
        elif opt in ("-t", "--testdir"):
            test_labels_dir = arg
        elif opt in ("-d", "--detectdir"):
            detect_labels_dir = arg
        elif opt in ("-o", "--ofile"):
            discardfilepath = arg

    main(test_labels_dir, detect_labels_dir, discardfilepath)