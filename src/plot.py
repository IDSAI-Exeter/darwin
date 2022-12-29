# TBC
import os

import pandas


def main(experiment_dir, n_augment, timings):

    k = 10

    l = []

    for i in range(1, k+1):
        print(experiment_dir + "fold_%i"%(n_augment+i))
        l += [('t', "%sfold_%i/raw_1/runs/raw/results.csv"%(experiment_dir, 10+i), 10+i)]
        l += [('a', "%sfold_%i/augment_1_%i/runs/augment/results.csv"%(experiment_dir, 10+i, n_augment), 10+i)]

    for t, f, i in l:
        os.system("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/fold_%s_%i.csv"%(f, t, i))
        print("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/fold_%s_%i.csv"%(f, t, i))

    import pandas as pd
    import matplotlib.pyplot as plt

    dfs = []
    i_ = [10 + i for i in range(1, k+1)]

    for i in i_:
        try:
            dfs.append(('raw_', pd.read_csv("csv/fold_t_%i.csv"%i), i))
            dfs.append(('aug_', pd.read_csv("csv/fold_a_%i.csv"%i), i))
        except:
            pass
    print(dfs)
    def cs(secs, df):
        ts = [secs]*len(df)
        import numpy as np
        cs = np.cumsum(ts)
        cs = list(cs)
        return cs

    i = 0
    while i < len(dfs):
        dfs[i][1]['cumtime'] = cs(timings[0], dfs[i][1])  # 100 raw
        dfs[i+1][1]['cumtime'] = cs(timings[1], dfs[i+1][1])  # 100 aug
        i += 2

    # Proper values to be read from log files.

    # dfs[0][1]['cumtime'] = cs(timings[0], dfs[0][1])  # 100 raw
    # dfs[1][1]['cumtime'] = cs(timings[1], dfs[1][1])  # 100 aug
    #
    # # try:
    # dfs[2][1]['cumtime'] = cs(timings[2], dfs[2][1])  # 500 raw
    # dfs[3][1]['cumtime'] = cs(timings[3], dfs[3][1])  # 500 aug
    #
    # dfs[4][1]['cumtime'] = cs(timings[4], dfs[4][1])  # 1000 raw
    # dfs[5][1]['cumtime'] = cs(timings[5], dfs[5][1])  # 1000 aug
    #
    # dfs[6][1]['cumtime'] = cs(timings[6], dfs[6][1])  # 1000 raw
    # dfs[7][1]['cumtime'] = cs(timings[7], dfs[7][1])  # 1000 aug
    #
    # dfs[8][1]['cumtime'] = cs(timings[0], dfs[8][1])  # 1000 raw
    # dfs[9][1]['cumtime'] = cs(timings[1], dfs[9][1])  # 1000 aug

    # except:
    #     pass

    fig, ax = plt.subplots()
    for t, df, i in dfs:#[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df.plot(ax=ax, x='cumtime', y=t+str(i), legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
        print(df.columns)
    plt.ylabel('metrics/mAP_0.5:0.95')
    plt.xlabel('time(s)')
    fig.savefig('time.png')

    fig, ax = plt.subplots()
    for t, df, i in dfs:#[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df[t+str(i)].plot(legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
    plt.xlabel('epochs')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('epochs.png')

    kfold = {}
    kfold['raw_'] = pandas.DataFrame()
    kfold['aug_'] = pandas.DataFrame()
    fig, ax = plt.subplots()
    for t, df, i in dfs:#[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        # df[t+str(i)].plot(legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
        kfold[t][i] = df['metrics/mAP_0.5:0.95']

    kfold['raw_']['raw'] = kfold['raw_'].mean(axis=1)
    kfold['aug_']['raw+aug'] = kfold['aug_'].mean(axis=1)
    kfold['raw_']['std'] = kfold['raw_'].std(axis=1)
    kfold['aug_']['std'] = kfold['aug_'].std(axis=1)

    kfold['raw_']['cumtime'] = cs(timings[0], kfold['raw_'])
    kfold['aug_']['cumtime'] = cs(timings[1], kfold['aug_'])

    # kfold['raw_'].plot(ax=ax, use_index=True, y='raw', color='black', title='K-fold cross validation')
    # kfold['aug_'].plot(ax=ax, use_index=True, y='raw+aug', color='gray')
    kfold['raw_'].plot(ax=ax, x='cumtime', y='raw', linestyle='solid', color='black') #, title='MonteCarlo Shuffle split cross validation augmented with %i images per species'%n_augment)
    kfold['aug_'].plot(ax=ax, x='cumtime', y='raw+aug', linestyle='dashdot', color='black')

    import math

    # plt.fill_between(x=kfold['aug_'].index, y1=kfold['aug_']['raw+aug'] - kfold['aug_']['std'], y2=kfold['aug_']['raw+aug'] + kfold['aug_']['std'])
    # plt.fill_between(x=kfold['raw_'].index, y1=kfold['raw_']['raw'] - kfold['raw_']['std'], y2=kfold['raw_']['raw'] + kfold['raw_']['std'])
    plt.fill_between(color='lightgray', x=kfold['aug_']['cumtime'], y1=kfold['aug_']['raw+aug'] - kfold['aug_']['std']/(2*math.sqrt(k)), y2=kfold['aug_']['raw+aug'] + kfold['aug_']['std']/(2*math.sqrt(k)))
    plt.fill_between(color='lightgray', x=kfold['raw_']['cumtime'], y1=kfold['raw_']['raw'] - kfold['raw_']['std']/(2*math.sqrt(k)), y2=kfold['raw_']['raw'] + kfold['raw_']['std']/(2*math.sqrt(k)))

    yerr = kfold['raw_']['std']

    # plt.xlabel('epochs')
    plt.xlabel('time(s)')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('montecarlo-shuffle.png')

    print(kfold)


if __name__ == "__main__":
    #timestamps = [13, 141, 134, 16, 108, 25, 111]
    #main('projects/darwin/data/experiments/ewg/', 200, timestamps)
    timings = [13*60+8, 33*60+7, 13*60+8, 33*60+7, 13*60+8, 33*60+7, 13*60+8, 33*60+7]
    timings = [35, 41]
    main('projects/darwin/data/experiments/montecarlo/', 1, timings)
    os.system("mv montecarlo-shuffle.png epochs.png time.png ../plots/")
