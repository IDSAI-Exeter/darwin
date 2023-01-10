# python3 matrix.py -e projects/darwin/data/experiments/montecarlo/ -a 1,2,4,8 -r 1,2,4,8

import glob
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn
import json

config = json.load(open('../config.json'))

plt.rcParams["figure.figsize"] = (10, 7)


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


def main(dir, raw_sizes, aug_factors, download=False, k = 1):

    try:
        os.mkdir("results/")
    except:
        pass
    if download:
        os.system("rm -rf results/")
        os.system(
            "scp -r -i ~/.ssh/id_rsa_jade %s@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s ."%(config['jade_account'], dir))

    dir = "results/"
    raw = None
    deltas = []
    species_list = None
    dfs = {}
    deltas_matrix = [[] for r in raw_sizes]
    deltas_species = {(r, a): [] for r in raw_sizes for a in aug_factors}

    for fold in ["fold_" + str(i) for i in range(11, 11+k)]:
        r_index = 0
        for j in raw_sizes:
            r = "raw_" + str(j)
            print(r, r_index)
            # for r in ["raw_" + str(i) for i in raw_sizes]:
            raw = parse(dir + "%s_%s.out"%(fold, r))
            if raw is not None:
                delta = []
                for a in aug_factors:
                    test = "augment_%i_"%j + str(a)
                    print(test)
                    augment = parse(dir + "%s_%s.out"%(fold, test))
                    if augment is not None:
                        print((j, a))
                        delta.append(float(augment.iloc[0]['mAP50-95']) - float(raw.iloc[0]['mAP50-95']))
                        # if len(augment) == 46:
                        species = []
                        for i in range(1, len(augment)):
                            # species_list.append(augment.iloc[i]['Class'])
                            species.append(float(augment.iloc[i]['mAP50-95']) - float(raw.iloc[i]['mAP50-95']))
                        deltas_species[(j, a)].append(species)
                        dfs[(j, a)] = augment
                deltas.append(delta)
                deltas_matrix[r_index].append(delta)
                # print(deltas_matrix)
            r_index += 1

    matrix = pd.DataFrame()
    std = pd.DataFrame()

    for r in range(len(raw_sizes)):
        df = pd.DataFrame(deltas_matrix[r])
        df.columns = [str(i) for i in aug_factors]
        # print("mean mAP delta r=%i"%r, df.mean())
        matrix[raw_sizes[r]] = df.mean()
        std[raw_sizes[r]] = df.std()/math.sqrt(k)

    print(matrix)
    print(std)

    import seaborn as sns
    # RdYlGn
    ax = sns.heatmap(matrix, annot=True, fmt=".7f", center=0, cmap="Spectral", cbar_kws={'label': "mean mAP delta over %i fold(s)"%k})
    ax.set(xlabel="# raw images per species", ylabel="augmentation factor")
    plt.savefig('../plots/matrix.png')

    plt.clf()
    # import matplotlib as mpl
    # norm = mpl.colors.Normalize(vmin=0.001, vmax=0.007)
    # RdYlGn
    ax = sns.heatmap(std, annot=True, fmt=".7f", center=0, cmap="gray_r", cbar_kws={'label': "standard error of the mean mAP delta over %i fold(s)"%k})
    ax.set(xlabel="# raw images per species", ylabel="augmentation factor")
    plt.savefig('../plots/matrix-std.png')
    # plt.show()

    # exit()

    # df = pd.DataFrame(deltas)
    # print(df)
    # df.columns = [str(i) for i in aug_factors]
    # print(df)
    # summary = pd.DataFrame()
    # summary['mean'] = df.mean()
    # summary['std err'] = df.std()/math.sqrt(len(df))

    # print(summary)

    species_columns = [(r, a) for r in raw_sizes for a in aug_factors]
    df_species = pd.DataFrame()

    for (r, a) in deltas_species.keys():
        # print((r, a))
        species_list = list(dfs[(r, a)]['Class'])[1:]
        df = pd.DataFrame(deltas_species[(r, a)])
        df.columns = species_list
        # print(df)
        mean = pd.DataFrame(df.mean(axis=0))
        mean.columns = [str((r, a))]
        df_species = pd.merge(df_species, mean, how='outer', left_index = True, right_index = True)

    print(df_species)
    plt.clf()
    # sns.set(rc={'figure.figsize': (80, 80)})
    matplotlib.rc('ytick', labelsize=8)
    ax = sns.heatmap(df_species, xticklabels=1, yticklabels=1,  annot=False, fmt=".7f", center=0, cmap="Spectral", cbar_kws={'label': "mean mAP delta over %i fold(s)"%k})
    ax.set(xlabel="(# raw images per species, augmentation factor)", ylabel="species")
    plt.tight_layout()
    plt.savefig('../plots/species.png')

    # transpose = df_species.T
    # transpose.columns = deltas_species_columns
    # print(transpose)
    # print(df_species.mean(axis=0))


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    aug_factors = [1, 2, 4]
    raw_sizes = [1]
    k = 1
    try:
        opts, args = getopt.getopt(argv, "he:a:r:k:", ["experiment_dir=", "aug_factors=", "raw_sizes=", "k-groups="])
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
        elif opt in ("-k", "--k-groups"):
            k = int(arg)

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    # main("../data/experiments/montecarlo/results/", [1], [1, 2, 4])
    main(experiment_dir + "results/", raw_sizes, aug_factors, True, k)
    os.system("git add ../plots/matrix-std.png ../plots/matrix.png ../plots/species.png;git commit -m 'test results update';git push")