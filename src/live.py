# TBC
import os


def main(experiment_dir, n_augment, timestamps):

    k = 4

    l = []

    for i in range(1, k+1):
        print(experiment_dir + "fold_%i"%(n_augment+i))
        l += [('t', "%sfold_%i/train/runs/train/results.csv"%(experiment_dir, 100+i), 100+i)]
        l += [('a', "%sfold_%i/augment_%i/runs/augment/results.csv"%(experiment_dir, 100+i, n_augment), 100+i)]

    for t, f, i in l:
        os.system("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/fold_%s_%i.csv"%(f, t, i))
        print("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/fold_%s_%i.csv"%(f, t, i))

    import pandas as pd
    import matplotlib.pyplot as plt

    dfs = []
    i_ = [100 + i for i in range(1, k+1)]

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

    # Proper values to be read from log files.

    dfs[0][1]['cumtime'] = cs(timestamps[0], dfs[0][1])  # 100 raw
    dfs[1][1]['cumtime'] = cs(timestamps[1], dfs[1][1])  # 100 aug

    # try:
    dfs[2][1]['cumtime'] = cs(timestamps[2], dfs[2][1])  # 500 raw
    dfs[3][1]['cumtime'] = cs(timestamps[3], dfs[3][1])  # 500 aug

    # dfs[4][1]['cumtime'] = cs(timestamps[4], dfs[4][1])  # 1000 raw
    # dfs[5][1]['cumtime'] = cs(timestamps[5], dfs[5][1])  # 1000 aug

    # dfs[6][1]['cumtime'] = cs(timestamps[6], dfs[6][1])  # 1000 raw
    # dfs[7][1]['cumtime'] = cs(timestamps[7], dfs[7][1])  # 1000 aug
     # except:
     #     pass

    fig, ax = plt.subplots()
    for t, df, i in dfs[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df.plot(ax=ax, x='cumtime', y=t+str(i), legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
        print(df.columns)
    plt.ylabel('metrics/mAP_0.5:0.95')
    plt.xlabel('time(s)')
    fig.savefig('time.png')

    fig, ax = plt.subplots()
    for t, df, i in dfs[:4]:
        df[t+str(i)] = df['metrics/mAP_0.5:0.95']
        df[t+str(i)].plot(legend=True, title='Superbeast 101 augmented with %i images per species'%n_augment)
    plt.xlabel('epochs')
    plt.ylabel('metrics/mAP_0.5:0.95')
    fig.savefig('epochs.png')


if __name__ == "__main__":
    #timestamps = [13, 141, 134, 16, 108, 25, 111]
    #main('projects/darwin/data/experiments/ewg/', 200, timestamps)
    timestamps = [13*60+8, 33*60+7, 13*60+8, 33*60+7, 13*60+8, 33*60+7]
    main('projects/darwin/data/experiments/all/', 100, timestamps)
    os.system("mv epochs.png time.png ../plots/")
