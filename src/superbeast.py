import os

def run(command):
    print("\t" + command)
    #os.system(command)


def main(experiment_dir, species, n_augment):

    k = 4

    print("\nGenerating %i FOLD experiment in %s for the following species :\n\t%s\n"%(k, experiment_dir, ', '.join(species)))

    j = n_augment

    try:
        os.mkdir(experiment_dir)
    except:
        pass

    for i, f in [(100, f) for f in range(1, k+1)]:

        fold_dir = experiment_dir + "fold_%i"%(i+f)

        try:
            os.mkdir(fold_dir)
        except:
            pass

        run("python3 experiment.py -e %s --n_images=%i --species=%s"%(fold_dir, i, ','.join(species)))
        run("python3 download_empty.py -e %s"%fold_dir)
        run("python3 augment.py -e %s -i %i"%(fold_dir, j))
        run("cd %s; sbatch sbatch_train.sh; sbatch sbatch_augment_%i.sh; cd -;"%(fold_dir, j))


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    n_augment = 0
    argv = sys.argv[1:]
    species = []

    try:
        opts, args = getopt.getopt(argv, "he:s:n:", ["experiment_dir=", "species=", "n_augment="])
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
        elif opt in ("-n", "--n_augment"):
            n_augment = int(arg)

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'

    main(experiment_dir, species, n_augment)

