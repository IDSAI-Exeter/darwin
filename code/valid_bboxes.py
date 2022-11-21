import json
import cv2

def main():

    json_data = None
    with open('../data/bbox_species.json') as json_file:
        json_data = json.load(json_file)
        json_file.close()

    season = 'S5' #S06
    location = 'C11' #'G07'

    seasons = sorted(list(set([annot['annotation']['season'] for annot in json_data])))

    discard = []
    keep = []

    cnt = 0

    # undo 'z'

    bins = []

    for season in seasons:
        data = [annot for annot in json_data if annot['annotation']['season'] == season ]
        locations = list(set([annot['annotation']['location'] for annot in data]))

        for location in locations:
            bins.append({'season': season, 'location': location})


    seen = []

    while bins:
        bin = bins[0]
        bins = bins[1:]
        season = bin['season']
        location = bin['location']
        seen.append({'season': season, 'location': location})


        print(len(seen), '/', len(bins), season, location)

        location_bboxes = [annot for annot in json_data if annot['annotation']['location'] == location and annot['annotation']['season'] == season]
        #image = cv2.imread('../data/serengeti_bboxes/images/'+image_id + '.JPG')
        images = list(set([bbox['image_id'] for bbox in location_bboxes]))

        validated = False

        while not validated:

            for image_id in images:
                image = cv2.imread('../data/serengeti_bboxes/images/'+image_id.split('/')[-1] + '.JPG')
                bboxes = [bbox for bbox in location_bboxes if bbox['image_id'] == image_id]

                for bbox in bboxes:
                    x, y, w, h = bbox['bbox']
                    #image = cv2.imread('../data/serengeti_bboxes/images/'+bbox['image_id'].split('/')[-1] + '.JPG')
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 0), 3)
                try:
                    cv2.imshow('image', image)
                except:
                    continue
                    print('file error')

                key = cv2.waitKey(0)
                # print(key)
                if key == 32:
                    continue
                elif key == 107:
                    # keep
                    keep.append({'season': season, 'location': location})
                    validated = True
                    break
                elif key == 100:
                    # discard
                    discard.append({'season': season, 'location': location})
                    validated = True
                    break
                elif key == 122:
                    # undo
                    last = seen.pop()
                    last = seen.pop()
                    try:
                        discard.remove(last)
                    except:
                        pass
                    try:
                        keep.remove(last)
                    except:
                        pass
                    bins = [last] + bins
                    validated = True
                    break

        #print('keep', keep)
        #print('discard', discard)

        json.dump(discard, open('discard.json', 'w'))
        json.dump(keep, open('keep.json', 'w'))


if __name__=='__main__':
    main()