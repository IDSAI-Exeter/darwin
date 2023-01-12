# TBC
import os
import pandas
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 7)
import json
config = json.load(open('../config.json'))


def main(experiment_dir, k, aug_factors, timings, raw_size, download=True):

    l = []

    if download:
        os.system("rm -rf csv/")
        os.mkdir("csv")
        for i in range(1, k+1):
            print(experiment_dir + "fold_%i"%(10+i))
            l += [('raw', "%sfold_%i/raw_%i/runs/raw/results.csv"%(experiment_dir, 10+i, raw_size), 10+i)]
            l += [('raw', "%sfold_%i/raw_%i/runs/raw2/results.csv"%(experiment_dir, 10+i, raw_size), 10+i)]
            for j in aug_factors:
                l += [('a_%i'%j, "%sfold_%i/augment_%i_%i/runs/augment/results.csv"%(experiment_dir, 10+i, raw_size, j), 10+i)]
                l += [('a_%i'%j, "%sfold_%i/augment_%i_%i/runs/augment2/results.csv"%(experiment_dir, 10+i, raw_size, j), 10+i)]

        for t, f, i in l:
            print("scp -i ~/.ssh/id_rsa_jade %s@jade2.hartree.stfc.ac.uk:%s/%s csv/fold_%i_r_%i_%s.csv"%(config['jade_account'], config['jade_home'], f, i, raw_size, t))
            os.system("scp -i ~/.ssh/id_rsa_jade %s@jade2.hartree.stfc.ac.uk:%s/%s csv/fold_%i_r_%i_%s.csv"%(config['jade_account'], config['jade_home'],  f, i, raw_size, t))

    dfs = []
    i_ = [10 + i for i in range(1, k+1)]

    for i in i_:
        # try:
        dfs.append(('raw_%i'%raw_size, pd.read_csv("csv/fold_%i_r_%i_raw.csv"%(i, raw_size)).iloc[0:300], i))
        for j in aug_factors:
            dfs.append(('aug_%i'%j, pd.read_csv("csv/fold_%i_r_%i_a_%i.csv"%(i, raw_size, j)).iloc[0:300], i))
        # except:
        #     pass
    print(dfs)
    def cs(secs, df):
        ts = [secs]*len(df)
        import numpy as np
        cs = np.cumsum(ts)
        cs = list(cs)
        return cs

    i = 0
    while i < len(dfs):
        print(i, len(dfs))
        dfs[i][1]['cumtime'] = cs(timings[0], dfs[i][1])  # 100 raw
        for j in range(1, len(aug_factors) + 1):
            print(i+j, len(dfs))
            dfs[i+j][1]['cumtime'] = cs(timings[j], dfs[i+j][1])  # 100 aug
        i += len(aug_factors) + 1

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
        df.plot(ax=ax, x='cumtime', y=t+str(i), legend=True, title='Superbeast 101')
        print(df.columns)
    plt.ylabel('metrics/mAP_0.5:0.95')
    plt.xlabel('time(s)')
    fig.savefig('time.png')

    fig, ax = plt.subplots()
    for t, df, i in dfs:#[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df[t+str(i)].plot(legend=True, title='Superbeast 101')
    plt.xlabel('epochs')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('epochs.png')

    kfold = {}
    kfold['raw_%i'%raw_size] = pandas.DataFrame()
    fig, ax = plt.subplots()

    for i in aug_factors:
        kfold['aug_%i'%i] = pandas.DataFrame()

    for t, df, i in dfs:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        # df[t+str(i)].plot(legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
        # print(t)
        kfold[t][i] = df['metrics/mAP_0.5:0.95']
    print(kfold)
    kfold['raw_%i'%raw_size]['raw_%i'%raw_size] = kfold['raw_%i'%raw_size].mean(axis=1)
    kfold['raw_%i'%raw_size]['std'] = kfold['raw_%i'%raw_size].std(axis=1)
    kfold['raw_%i'%raw_size]['cumtime'] = cs(timings[0], kfold['raw_%i'%raw_size])

    print(kfold)
    import math
    # kfold['raw_%i'%raw_size].plot(ax=ax, use_index=True, y='raw', color='black', title='K-fold cross validation')
    # kfold['aug_'].plot(ax=ax, use_index=True, y='raw+aug', color='gray')

    kfold['raw_%i'%raw_size].plot(ax=ax, x='cumtime', y='raw_%i'%raw_size, linestyle='solid', color='black', title='%i-fold Monte Carlo cross validation mean mAP per epoch'%k)
    plt.fill_between(color='lightgray', x=kfold['raw_%i'%raw_size]['cumtime'], y1=kfold['raw_%i'%raw_size]['raw_%i'%raw_size] - kfold['raw_%i'%raw_size]['std']/(2*math.sqrt(k)), y2=kfold['raw_%i'%raw_size]['raw_%i'%raw_size] + kfold['raw_%i'%raw_size]['std']/(2*math.sqrt(k)))

    # linestyles = ['dashed', 'dashdot', 'dotted', '--']

    colors = [(0.9, 0.6, 0), (0.33, 0.7, 0.9), (0, 0.6, 0.45), (0.83, 0.36, 0)]

    # styles = {1 : 0, 2 : 1, 4 : 2, 8 : 3}
    styles = {i: j for i, j in zip(aug_factors, range(len(aug_factors)))}

    for j in aug_factors:
        kfold['aug_%i'%j]['raw_%i+aug_%i'%(raw_size,j)] = kfold['aug_%i'%j].mean(axis=1)
        kfold['aug_%i'%j]['std'] = kfold['aug_%i'%j].std(axis=1)
        kfold['aug_%i'%j]['cumtime'] = cs(timings[1], kfold['aug_%i'%j])
        # print(kfold['aug_%i'%j])
        # , linestyle = linestyles[styles[j]]
        aug = kfold['aug_%i'%j]#.iloc[:45]
        aug.plot(ax=ax, x='cumtime', y='raw_%i+aug_%i'%(raw_size, j), color=colors[styles[j]])
        plt.fill_between(color='lightgray', x=aug['cumtime'], y1=aug['raw_%i+aug_%i'%(raw_size, j)] - aug['std']/(2*math.sqrt(k)), y2=aug['raw_%i+aug_%i'%(raw_size, j)] + aug['std']/(2*math.sqrt(k)))


    # plt.fill_between(x=kfold['aug_'].index, y1=kfold['aug_']['raw+aug'] - kfold['aug_']['std'], y2=kfold['aug_']['raw+aug'] + kfold['aug_']['std'])
    # plt.fill_between(x=kfold['raw_%i'%raw_size].index, y1=kfold['raw_%i'%raw_size]['raw'] - kfold['raw_%i'%raw_size]['std'], y2=kfold['raw_%i'%raw_size]['raw'] + kfold['raw_%i'%raw_size]['std'])

    # yerr = kfold['raw_%i'%raw_size]['std']

    # plt.xlabel('epochs')
    plt.xlabel('time(s)')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('montecarlo-shuffle.png')

    delta = {}
    # delta['raw_%i'%raw_size] = pandas.DataFrame()
    # fig, ax = plt.subplots()

    # for t, df, i in dfs: #[:4]:
        # print(t, i, df)

    raw = kfold['raw_%i'%raw_size]

    for j in aug_factors:
        delta['aug_%i'%j] = pandas.DataFrame()
        for index, aug in kfold['aug_%i' % j].iterrows():
            v = aug['raw_%i+aug_%i'%(raw_size,j)]
            t = aug['cumtime']
            v_raw = None
            for index, row in raw.iterrows():
                t2 = row['cumtime']
                if t2 > t:
                    break
                v_raw = row['raw_%i'%raw_size]
            if v_raw is not None:
                d = v - v_raw
                delta['aug_%i' % j] = pandas.concat([delta['aug_%i' % j], pandas.DataFrame([{'cumtime': t, 'mAP delta': d}])])

    fig, ax = plt.subplots()
    for k in delta.keys():
        # print(delta[k])
        delta[k].plot(ax=ax, x='cumtime', y='mAP delta', legend=True, title='mAP Delta')
    plt.xlabel('time(s)')
    plt.ylabel('metrics/mAP_0.5:0.95 delta')
    fig.savefig('delta_%i.png'%raw_size)

    # print(kfold)


if __name__ == "__main__":
    #timestamps = [13, 141, 134, 16, 108, 25, 111]
    #main('projects/darwin/data/experiments/ewg/', 200, timestamps)

    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    k = 1

    try:
        opts, args = getopt.getopt(argv, "he:k:", ["experiment_dir=", "k-groups="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-k", "--k-groups"):
            k = int(arg)

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    # timings = [13*60+8, 33*60+7, 13*60+8, 33*60+7, 13*60+8, 33*60+7, 13*60+8, 33*60+7]
    # timings = [35, 41]
    # main('projects/darwin/data/experiments/montecarlo_/', 1, timings)
    # main(experiment_dir, 1, timings)

    timings = [35, 44, 52, 67, 80]

    timings = []
    # timings raw 1
    timings.append([35, 44, 52, 67, 120])
    # timings raw 2
    timings.append([60, 76, 93, 130, 197])
    # timings raw 4
    timings.append([81, 117, 151, 225, 369])
    # timings raw 8
    timings.append([120, 188, 261, 407, 682])

    # raw_size = 8

    raw_sizes = [1, 2, 4, 8]
    aug_factors = [1, 2, 4, 8]

    # raw_sizes = [500]
    # aug_factors = [1]
    # timings = [[780, 5340]]

    for i in range(0, len(raw_sizes)):
        raw_size = raw_sizes[i]
        main(experiment_dir, k, aug_factors, timings[i], raw_size, download=True)
        os.system("mv montecarlo-shuffle.png epochs.png time.png ../plots/")
        os.system("cp ../plots/montecarlo-shuffle.png ../plots/montecarlo_%i.png"%raw_size)
        # os.system("git add ../plots/montecarlo_*;git commit -m 'training results update'; git push")