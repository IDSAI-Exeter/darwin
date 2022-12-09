import os

l = ["projects/darwin/data/experiments/elephant/train/runs/train/results.csv"]
l += ["projects/darwin/data/experiments/elephant/augment_%i/runs/augment/results.csv"%i for i in [500, 1000, 2000, 4000, 8000]]

for f, s in zip(l, [0, 500, 1000, 2000, 4000, 8000]):
    os.system("scp -i ~/.ssh/id_rsa_jade ccm30-dxa01@jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD013/dxa01/ccm30-dxa01/%s csv/%i.csv"%(f,s))

import pandas as pd

import matplotlib.pyplot as plt

dfs = []
i_ = [0, 500, 1000, 2000, 4000, 8000]

for i in i_:
    dfs.append(pd.read_csv("csv/%i.csv"%i))

for df, i in zip(dfs, i_):
    print(df.columns)
    #df[str(i)] = df['     metrics/mAP_0.5']
    df[str(i)] = df['metrics/mAP_0.5:0.95']
    df[str(i)].plot(legend=True, title='Elephant detection starting with 1000 training images')

#plt.show()

plt.xlabel('epochs')
plt.ylabel('metrics/mAP_0.5:0.95')
plt.savefig('elephant.png')

