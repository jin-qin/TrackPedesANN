import os
import glob
import json
from scipy.io import loadmat
import cv2 as cv
from collections import defaultdict

class CaltechLoader:

    def __init__(self, root_dir):

        self.input_dir = os.path.join(root_dir, "caltech")
        self.output_dir = os.path.join(self.input_dir, "cached")

        # if no data has been generated yet, generate it once
        # TODO remove "True or"
        if True or not os.path.exists(self.output_dir):
            print("No cached dataset has been found. Generate it once.")
            self.loadAnnotations()
            self.loadImages()
        else:
            print("Cached caltech dataset has been found.")


    # loads all annotations to self.annotations
    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_annotations.py
    def loadAnnotations(self):

        all_obj = 0
        self.annotations = defaultdict(dict)
        for dname in sorted(glob.glob(os.path.join(self.input_dir, 'annotations/set*'))):
            set_name = os.path.basename(dname)
            self.annotations[set_name] = defaultdict(dict)
            for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
                vbb = loadmat(anno_fn)
                nFrame = int(vbb['A'][0][0][0][0][0])
                objLists = vbb['A'][0][0][1][0]
                maxObj = int(vbb['A'][0][0][2][0][0])
                objInit = vbb['A'][0][0][3][0]
                objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
                objStr = vbb['A'][0][0][5][0]
                objEnd = vbb['A'][0][0][6][0]
                objHide = vbb['A'][0][0][7][0]
                altered = int(vbb['A'][0][0][8][0][0])
                log = vbb['A'][0][0][9][0]
                logLen = int(vbb['A'][0][0][10][0][0])

                video_name = os.path.splitext(os.path.basename(anno_fn))[0]
                self.annotations[set_name][video_name]['nFrame'] = nFrame
                self.annotations[set_name][video_name]['maxObj'] = maxObj
                self.annotations[set_name][video_name]['log'] = log.tolist()
                self.annotations[set_name][video_name]['logLen'] = logLen
                self.annotations[set_name][video_name]['altered'] = altered
                self.annotations[set_name][video_name]['frames'] = defaultdict(list)

                n_obj = 0
                for frame_id, obj in enumerate(objLists):
                    if len(obj) > 0:
                        for id, pos, occl, lock, posv in zip(
                                obj['id'][0], obj['pos'][0], obj['occl'][0],
                                obj['lock'][0], obj['posv'][0]):
                            keys = obj.dtype.names
                            id = int(id[0][0]) - 1  # MATLAB is 1-origin
                            pos = pos[0].tolist()
                            occl = int(occl[0][0])
                            lock = int(lock[0][0])
                            posv = posv[0].tolist()

                            datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                            datum['lbl'] = str(objLbl[datum['id']])
                            datum['str'] = int(objStr[datum['id']])
                            datum['end'] = int(objEnd[datum['id']])
                            datum['hide'] = int(objHide[datum['id']])
                            datum['init'] = int(objInit[datum['id']])
                            self.annotations[set_name][video_name][
                                'frames'][frame_id].append(datum)
                            n_obj += 1

                print(set_name, video_name, n_obj)
                all_obj += n_obj

        print('Number of objects:', all_obj)

        # TODO save as json?
        #json.dump(self.annotations, open(os.path.join(self.output_dir, '/annotations.json', 'w')))


    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py
    def loadImages(self):

        print("extracting frames")

        # create output folder if it doesn't exist yet
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        imagesets = sorted(glob.glob(os.path.join(self.input_dir, 'set*')))
        print(len(imagesets), " extracted image sets found")

        for dname in imagesets:
            set_name = os.path.basename(dname)
            for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
                video_name = os.path.splitext(os.path.basename(fn))[0]
                cap = cv.VideoCapture(fn)
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    self.extractPedestriansFromImage(frame, set_name, video_name, i)

                    # TODO save original raw image on disk?
                    #self.save_img(dname, fn, i, frame)

                    i += 1
                print(fn)


    def save_img(self, dname, fn, i, frame):
        cv.imwrite('{}/{}_{}_{}.png'.format(
            self.output_dir, os.path.basename(dname),
            os.path.basename(fn).split('.')[0], i), frame)


    def extractPedestriansFromImage(self, img, set_name, video_name, frame_i):

        print("Extract pedestrians")

        # TODO when saving and loading json file before, we might need frame_i = str(frame_i)

        if frame_i in self.annotations[set_name][video_name]['frames']:
            data = self.annotations[set_name][
                video_name]['frames'][frame_i]
            for datum in data:
                x, y, w, h = [int(v) for v in datum['pos']]

                img_ped = img[y:y + h, x:x + w]
                self.save_img(set_name, video_name, frame_i, img_ped)

                #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                #n_objects += 1
            #wri.write(img)


    def getTrainingData(self):

        # TODO get training data
        Xall = None
        Yall = None
        return Xall, Yall


# TODO remove debug code
cfc_datasetpath = "/media/th/6C4C-2ECD/ml_datasets"
calLoader = CaltechLoader(cfc_datasetpath)