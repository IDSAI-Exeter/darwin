# move this to the generate_augmented_ds code for future datasets

def main(countscsv, augmenteddir, yamlfile, augmentedcsv):
    import pandas
    import yaml
    import glob

    df = pandas.read_csv(countscsv)
    dataset = yaml.load(open(yamlfile), Loader=yaml.FullLoader)

    classes = {v: k for k, v in dataset['names'].items()}
    names = dataset['names']
    labels_files = glob.glob(augmenteddir + '*.txt')

    from collections import Counter

    counts = Counter()
    for x in classes.keys():
        counts[str(x)] = 0

    for file in labels_files:
        label_f = open(file).readlines()
        bboxes = []
        for line in label_f:
            cl, x, y, w, h = line.split()
            counts[names[int(cl)]] += 1

    df.columns = ['id', 'name', 'total', 'train', 'test', 'val']

    for index, row in df.iterrows():
        df.at[index, 'aug'] = counts[row['name']]
        df.at[index, 'class'] = int(classes[row['name']])

    df.to_csv(augmentedcsv)

if __name__ == '__main__':
    import sys, getopt

    countscsv = ''
    augmenteddir = ''
    yamlfile = ''
    augmentedcsv = ''
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "hc:a:y:o:", ["countscsv=", "augmenteddir=", "yamlfile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -c <countscsv> -a <augmenteddir> -y <yamlfile> -o <augmentedcsv>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -c <countscsv> -a <augmenteddir> -y <yamlfile> -o <augmentedcsv>')
            sys.exit()
        elif opt in ("-c", "--countscsv"):
            countscsv = arg
        elif opt in ("-a", "--augmenteddir"):
            augmenteddir = arg
        elif opt in ("-y", "--yamlfile"):
            yamlfile = arg
        elif opt in ("-o", "--ofile"):
            augmentedcsv = arg

    main(countscsv, augmenteddir, yamlfile, augmentedcsv)
