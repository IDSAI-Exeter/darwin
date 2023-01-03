import glob
import os
import pandas as pd
import math

download = False


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

    try:
        os.mkdir(results)
    except:
        pass
    if download:
        os.system(
            "scp -r -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s results/"%dir)

    dir = "results/"
    raw = None
    deltas = []
    deltas_species = []
    species_list = None
    dfs = []
    for fold in ["fold_" + str(i) for i in range(11, 21)]:
        delta = []
        for j in raw_sizes:
            r = "raw_" + str(j)
            # for r in ["raw_" + str(i) for i in raw_sizes]:
            raw = parse(dir + "%s_%s.out"%(fold, r))
            if raw is not None:
                dfs.append(raw)
                for test in ["augment_%i_"%j + str(i) for i in aug_factors]:
                    augment = parse(dir + "%s_%s.out"%(fold, test))
                    if augment is not None:
                        delta.append(float(augment.iloc[0]['mAP50-95']) - float(raw.iloc[0]['mAP50-95']))
                        if test == "augment_%i_1"%j and len(augment) == 46:
                            species = []
                            for i in range(1, len(augment)):
                                # species_list.append(augment.iloc[i]['Class'])
                                species.append(float(augment.iloc[i]['mAP50-95']) - float(raw.iloc[i]['mAP50-95']))
                            deltas_species.append(species)
                deltas.append(delta)

    df = pd.DataFrame(deltas)
    df.columns = [str(i) for i in aug_factors]
    print(df)
    summary = pd.DataFrame()
    summary['mean'] = df.mean()
    summary['std err'] = df.std()/math.sqrt(len(df))

    print(summary)

    species_list = list(dfs[0]['Class'])[1:]
    df_species = pd.DataFrame(deltas_species)
    # print(df_species)
    df_species.columns = species_list
    print(df_species.T)
    print(df_species.mean(axis=0))


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    aug_factors = [1, 2, 4]
    raw_sizes = [1]

    try:
        opts, args = getopt.getopt(argv, "he:a:r:", ["experiment_dir=", "aug_factors="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -a <aug_factors>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-a", "--aug_factors"):
            aug_factors = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-r", "--raw_sizes"):
            raw_sizes = [int(s.strip().lower()) for s in arg.split(',')]

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    # main("../data/experiments/montecarlo/results/", [1], [1, 2, 4])
    main(experiment_dir + "results/", raw_sizes, aug_factors)
