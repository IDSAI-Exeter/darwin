import glob
import pandas as pd

def parse(f):
    intable = False
    rows = []
    txt = open(f).readlines()
    for line in txt:
        for space in reversed(range(1,20)):
            line = line.strip().replace(" "*space, "\t")
        row = line.split("\t")
        if row[0] == 'all':
            intable = True
        if row[0] == 'Speed:':
            intable = False
        if intable:
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.columns = ['Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95']
        return df
    else:
        return None

def main(dir):
    raw = None
    deltas = []
    for fold in ["fold_" + str(i) for i in range(11, 21)]:
        delta = []
        for test in ["raw_" + str(i) for i in [1]]:
            raw = parse(dir + "%s_%s.out"%(fold, test))
            if raw:
                for test in ["augment_1_" + str(i) for i in [1, 2, 4, 8]]:
                    augment = parse(dir + "%s_%s.out"%(fold, test))
                    delta.append(float(augment.loc[0]['mAP50-95']) - float(raw.loc[0]['mAP50-95']))
                deltas.append(delta)

    df = pd.DataFrame(delta)
    print(df)

if __name__ == "__main__":
    main("../data/experiments/montecarlo/results/")
