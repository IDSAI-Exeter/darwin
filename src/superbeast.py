import os

def run(command):
    print("\t" + command)
    os.system(command)


def main(experiment_dir, species, raw_sizes, aug_factors, n_empty):

    k = 10

    print("\nGenerating %i FOLD experiment in %s for the following species :\n\t%s\n"%(k, experiment_dir, ', '.join(species)))

    try:
        os.mkdir("../data/experiments/")
    except:
        pass

    try:
        os.mkdir(experiment_dir)
    except:
        pass

    for i, f in [(10, f) for f in range(1, k+1)]:

        fold_dir = experiment_dir + "fold_%i"%(i+f)

        try:
            os.mkdir(fold_dir)
        except:
            pass

        run("python3 experiment.py -e %s --n_images=%i --species=%s"%(fold_dir, i, ','.join(species))) # n is the number of images per species for the validation set.
        run("python3 download_empty.py -e %s -n %i"%(fold_dir, n_empty))
        run("python3 augment.py -e %s -r %s -a %s --species=%s"%(fold_dir, ','.join([str(x) for x in raw_sizes]), ','.join([str(x) for x in aug_factors]), ','.join(species)))
        run("cd %s; for f in sbatch_*; do sbatch \"$f\"; done; cd -;"%fold_dir)


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    species = []
    aug_factors = [1, 2, 4]
    raw_sizes = [1, 5, 10]
    n_empty = 0

    try:
        opts, args = getopt.getopt(argv, "he:s:r:a:n:", ["experiment_dir=", "species=", "raw_sizes=", "aug_factors=", "n_empty="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -s <species> -n <n_augment>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -s <species> -n <n_augment>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-s", "--species"):
            species = [s.strip().lower() for s in arg.split(',')]
        elif opt in ("-r", "--raw_sizes"):
            raw_sizes = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-a", "--aug_factors"):
            aug_factors = [int(s.strip().lower()) for s in arg.split(',')]
        elif opt in ("-n", "--n_empty"):
            n_empty = int(arg)

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    main(experiment_dir, species, raw_sizes, aug_factors, n_empty)

