import glob
import pandas as pd
import math

def parse(f):
    intable = False
    rows = []
    try:
        txt = open(f).readlines()
    except:
        return None
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

def main(dir, raw_sizes, aug_factors):
    raw = None
    deltas = []
    deltas_species = []

    for fold in ["fold_" + str(i) for i in range(11, 21)]:
        delta = []
        species = []
        for r in ["raw_" + str(i) for i in raw_sizes]:
            raw = parse(dir + "%s_%s.out"%(fold, r))
            if raw is not None:
                for test in ["augment_1_" + str(i) for i in aug_factors]:
                    augment = parse(dir + "%s_%s.out"%(fold, test))
                    if augment is not None:
                        delta.append(float(augment.loc[0]['mAP50-95']) - float(raw.loc[0]['mAP50-95']))
                        if test == "augment_1_1":
                            for i in range(1, len(augment)):
                                species.append(float(augment.loc[i]['mAP50-95']) - float(raw.loc[i]['mAP50-95']))
                deltas_species.append(species)
                deltas.append(delta)

    df = pd.DataFrame(deltas)
    df.columns = [str(i) for i in aug_factors]
    print(df.mean())
    print(df.std()/math.sqrt(len(df)))

    df_species = pd.DataFrame(deltas_species)
    df_species.columns = list(raw['Class'])[1:]
    print(df_species)

if __name__ == "__main__":
    main("../data/experiments/montecarlo/results/", [1], [1, 2, 4])
