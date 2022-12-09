# TBC

def main(experiment_dir, n_augment):

    k = 4

    for i in range(1, k+1):
        experiment_dir + "fold_%i"%(n_augment+i)

    import os

    l = [('t', "projects/darwin/data/experiments/eg_%i/train/runs/train/results.csv"%i, i) for i in [100, 500, 1000]]
    l += [('a', "projects/darwin/data/experiments/eg_%i/augment_200/runs/augment/results.csv"%i, i) for i in [100, 500, 1000]]

    for t, f, i in l:
        os.system("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/eg_%s_%i.csv"%(f, t, i))


    import pandas as pd
    import matplotlib.pyplot as plt

    dfs = []
    i_ = [100, 500, 1000]

    for i in i_:
        dfs.append(('raw_', pd.read_csv("csv/eg_t_%i.csv"%i), i))
        dfs.append(('aug_', pd.read_csv("csv/eg_a_%i.csv"%i), i))


    def cs(secs, df):
        ts = [secs]*len(df)
        import numpy as np
        cs = np.cumsum(ts)
        cs = list(cs)
        return cs


    dfs[0][1]['cumtime'] = cs(11, dfs[0][1])  # 100 raw
    dfs[1][1]['cumtime'] = cs(441, dfs[1][1])  # 100 aug

    dfs[2][1]['cumtime'] = cs(159, dfs[2][1])  # 500 raw
    dfs[3][1]['cumtime'] = cs(685, dfs[3][1])  # 500 aug

    dfs[4][1]['cumtime'] = cs(337, dfs[4][1])  # 1000 raw
    dfs[5][1]['cumtime'] = cs(770, dfs[5][1])  # 1000 aug

    fig, ax = plt.subplots()
    for t, df, i in dfs:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df.plot(ax=ax, x='cumtime', y=t+str(i), legend=True, title='Elephant-Giraffe detection augmented with 1000 images')
        print(df.columns)
    plt.ylabel('metrics/mAP_0.5:0.95')
    plt.xlabel('time(s)')
    fig.savefig('eg_time.png')

    fig, ax = plt.subplots()
    for t, df, i in dfs:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df[t+str(i)].plot(legend=True, title='Elephant-Giraffe detection augmented with 1000 images')
    plt.xlabel('epochs')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('eg_epochs.png')

