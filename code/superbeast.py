import os

def run(command):
    print("\t" + command)
    os.system(command)

def main(experiment_dir, species):
    # K-FOLD
    k = 4
    print("\nGenerating %i FOLD experiment in %s for the following species :\n\t%s\n"%(k, experiment_dir, ', '.join(species)))

    j = 2000

    for i in [100]:
        run("python3 meta_experiment_reloaded.py -e %s --n_images=%i --species=%s"%(experiment_dir, i, ','.join(species)))
        run("python3 download_empty_reloaded.py -e %s"%experiment_dir)
        run("python3 augment_trainset_reloaded.py -e %s -i %i"%(experiment_dir, j))


if __name__ == "__main__":
    import sys, getopt

    experiment_dir = ''
    n_images = 0
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "he:s:", ["experiment_dir=", "species="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir> -s <species>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('script.py -e <experiment_dir> -s <species>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-s", "--species"):
            species = [s.strip().lower() for s in arg.split(',')]

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'
    main(experiment_dir, species)

