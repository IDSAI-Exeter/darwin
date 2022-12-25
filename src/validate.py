import os

for fold in ["fold_" + str(i) for i in range(11,21)]:
    for test in ["raw_1"] + ["augment_1_" + str(i) for i in [1,2,4,8]]:
        os.system("cd %s; python3 ../../../../lib/yolov5/val.py --data %s.yaml --weights %s/runs/augment/weights/best.pt --project validate/ --verbose --task test 2> ../results/%s_%s.out;cd .."%(fold, test, test, fold, test))

