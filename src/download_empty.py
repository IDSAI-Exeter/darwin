import json
import os


def main(experiment_dir, n_images_per_locations):
    locations_file = experiment_dir + 'test_locations.json'
    empty_directory = experiment_dir + '/empty/'

    try:
        os.mkdir(empty_directory)
    except:
        pass


    json_data = None

    # with open('../data/bbox_species.json') as json_file:
    #     json_data = json.load(json_file)
    #     json_file.close()

    with open('../data/SnapshotSerengeti_S1-11_v2.1.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    locations = None
    with open(locations_file) as json_file:
        locations = json.load(json_file)
        json_file.close()


    empty_images = []

    for location in locations:
        empty_images_at_location = list(set([annot['image_id'] for annot in json_data['annotations']
                                if annot['category_id'] == 0 and annot['location'] == location]))
        import random
        random.shuffle(empty_images_at_location)
        empty_images += empty_images_at_location[0:n_images_per_locations]


    batchsize = 100
    from_ = 0

    n = len(empty_images) #5000
    print(len(empty_images))

    for i in range(from_, n, batchsize):
        batch = empty_images[i:i+batchsize]
        open("download_file_list.txt", 'w').write(".JPG\n".join(batch))
        # --output-level quiet
        os.system("../lib/azcopy/azcopy cp  https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped/ '%s' --list-of-files download_file_list.txt --output-level quiet"%empty_directory)
        print('#remaining empty images', n - i-batchsize, '#next_i', i+batchsize)

    os.system("find %s -name '*.JPG' -exec mv {} %s \;"%(empty_directory+'snapshotserengeti-unzipped/', empty_directory))
    os.system("rm -rf %s"%(empty_directory+'snapshotserengeti-unzipped/'))

    print("next augment dataset")

if __name__=="__main__":
    import sys, getopt

    experiment_dir = ''
    argv = sys.argv[1:]
    n_images_per_locations = 10

    try:
        opts, args = getopt.getopt(argv, "he:", ["experiment_dir=", "n_images_per_locations="])
    except getopt.GetoptError:
        print('script.py -e <experiment_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -e <experiment_dir>')
            sys.exit()
        elif opt in ("-e", "--experiment_dir"):
            experiment_dir = arg
        elif opt in ("-n", "--n_images_per_locations"):
            n_images_per_locations = arg

    if not experiment_dir[-1] == '/':
        experiment_dir += '/'
    main(experiment_dir, n_images_per_locations)
