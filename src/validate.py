import os
import os.path


def main(dir, aug_factors, raw_sizes):
    try:
        os.mkdir("%sresults"%dir)
    except:
        pass
    for fold in ["fold_" + str(i) for i in range(11,21)]:
        for test in ["raw_" + str(i) for i in raw_sizes]:
            if not os.path.exists("%sresults/%s_%s.out"%(dir, fold, test)):
                os.system("cd %s; python3 ../../../../lib/yolov5/val.py --data %s.yaml --weights %s/runs/raw/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd -" % (dir + fold, test, test, fold, test))
        for test in ["augment_" + str(i) + "_" + str(j) for i in raw_sizes for j in aug_factors]:
            if not os.path.exists("%sresults/%s_%s.out"%(dir, fold, test)):
                os.system("cd %s; python3 ../../../../lib/yolov5/val.py --data %s.yaml --weights %s/runs/augment/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd .."%(dir + fold, test, test, fold, test))

    # for fold in [dir + "fold_" + str(i) for i in range(11,21)]:
    #     for test in ["raw_1"] + ["augment_1_" + str(i) for i in [1,2,4,8]]:
    # for fold in ["fold_" + str(i) for i in range(14,21)]:
    #     # fold = "fold_14"
    #     test = "raw_1"
    #     os.system("cd %s; python3 ../../../../lib/yolov5/val.py --data %s.yaml --weights %s/runs/raw/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd -"%(dir+fold, test, test, fold, test))


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    aug_factors = [1,2,4]
    raw_sizes = [1,2]
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "he:a:r:", ["experiment_dir="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -a <aug_factors>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -a <aug_factors>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-a", "--aug_factors"):
            aug_factors = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-r", "--raw_sizes"):
            raw_sizes = [int(s.strip().lower()) for s in arg.split(',')]

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    # main("../data/experiments/montecarlo/")
    main(experiment_dir, aug_factors, raw_sizes)
