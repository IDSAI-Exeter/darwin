import json

def main(rawdistribution, augmenteddistribution, plotfilename):
    raw = json.load(open(rawdistribution))
    aug = json.load(open(augmenteddistribution))

    raw_d = {}
    aug_d = {}

    results = []

    for x, y in zip(raw['x'], raw['y']):
        raw_d[x] = y

    for x, y in zip(aug['x'], aug['y']):
        aug_d[x] = y

    for x in raw['x']:
        results.append({'species': x,  'iou acc': raw_d[x], 'pred': 'raw'})
        results.append({'species': x,  'iou acc': aug_d[x], 'pred': 'aug'})

    import pandas
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    df = pandas.DataFrame(results)
    plt.rcParams["figure.figsize"] = (15, 10)
    sns.barplot(orient='h',
            x='iou acc',
            y='species',
            hue='pred',
            data=df)

    # Save the plot
    plt.tight_layout()
    plt.savefig(plotfilename)

if __name__ == '__main__':
    import sys, getopt

    rawdistribution = ''
    augmenteddistribution = ''
    plotfilepath = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "hr:a:o:", ["rawdistribution=", "augmenteddistribution=", "ofile="])
    except getopt.GetoptError:
        print('test.py -r <rawdistribution> -a <augmenteddistribution> -o <plotfilename>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -r <rawdistribution> -d <augmenteddistribution> -o <plotfilename>')
            sys.exit()
        elif opt in ("-r", "--rawdistribution"):
            rawdistribution = arg
        elif opt in ("-a", "--augmenteddistribution"):
            augmenteddistribution = arg
        elif opt in ("-o", "--ofile"):
            plotfilepath = arg

    main(rawdistribution, augmenteddistribution, plotfilepath)
