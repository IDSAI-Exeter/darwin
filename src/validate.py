import os

def main(dir):
    # for fold in [dir + "fold_" + str(i) for i in range(11,21)]:
    #     for test in ["raw_1"] + ["augment_1_" + str(i) for i in [1,2,4,8]]:
    #         os.system("cd %s; python3 ../../../../lib/yolov5/val.py --data %s.yaml --weights %s/runs/augment/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd .."%(fold, test, test, fold, test))

    # for fold in [dir + "fold_" + str(i) for i in range(11,21)]:
    #     for test in ["raw_1"] + ["augment_1_" + str(i) for i in [1,2,4,8]]:
    fold = "fold_14"
    test = "raw_1"
    os.system("cd %s; python3 ../lib/yolov5/val.py --data %s.yaml --weights %s/runs/augment/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd -"%(dir+fold, test, test, dir, fold, test))

if __name__=="__main__":
    main("../data/experiments/montecarlo/")
