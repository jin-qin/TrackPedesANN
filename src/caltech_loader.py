import os
import glob
import json
import gc
from scipy.io import loadmat
import cv2 as cv
from collections import defaultdict
import numpy as np
import cPickle as pickle
import scipy.stats as st

class CaltechLoader:

    def __init__(self, root_dir):

        self.image_width = 48
        self.image_height = 128
        self.image_size_min_resize = self.image_width / 2 # minimum length of minimum-length image length to allow resizing
        self.input_dir = os.path.join(root_dir, "caltech")
        self.output_dir = os.path.join(self.input_dir, "cached")

    def loadDataSet(self,training):

        if training:
            name = "caltech-training.p"
        else:
            name = "caltech-test.p"

        # if no data has been generated yet, generate it once
        cache_file = os.path.join(self.output_dir, name)
        if True or not os.path.exists(cache_file): #TODO remove True or to enable caching again
            print("No cached dataset has been found. Generate it once.")
            # TODO do we need to normalize the images to set the face to a specific position??
            self.loadAnnotations()
            self.loadImages(training)
        else:
            print("Cached caltech dataset has been found. Start loading..")

            if training:
                self.trainingSamplesPrevious, self.trainingSamplesCurrent = pickle.load(open(cache_file, "rb"))
            else:
                self.testSamplesPrevious, self.testSamplesCurrent = pickle.load(open(cache_file, "rb"))

            print("Finished cached data import.")

    # loads all annotations to self.annotations
    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_annotations.py
    def loadAnnotations(self):

        all_obj = 0
        self.annotations = defaultdict(dict)
        for dname in sorted(glob.glob(os.path.join(self.input_dir, 'annotations/set*'))):
            set_name = os.path.basename(dname)
            print("Annotations from set", set_name)
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


                all_obj += n_obj

        print('Number of objects:', all_obj)

        # TODO save as json?
        #json.dump(self.annotations, open(os.path.join(self.output_dir, '/annotations.json', 'w')))


    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py
    def loadImages(self, training): # TODO implement training parameter to support loading of test data, too

        print("extracting frames")

        # create output folder if it doesn't exist yet
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        imagesets = sorted(glob.glob(os.path.join(self.input_dir, 'set*')))


        image_folders_only = []
        for dname in imagesets:
            if os.path.isdir(dname):
                image_folders_only.append(dname)

        print(len(image_folders_only), " extracted image sets found")
        skipped = len(imagesets) - len(image_folders_only)
        if skipped > 0:
            print(skipped, " further files skipped. Forgot to extract?")

        self.trainingSamplesPrevious, self.trainingSamplesCurrent = [], []

        for dname in image_folders_only:
            set_name = os.path.basename(dname)

            print("Processing set ", set_name)

            for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
                video_name = os.path.splitext(os.path.basename(fn))[0]

                print("Processing video ", set_name, video_name)

                cap = cv.VideoCapture(fn)
                previousFrame = None
                i = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if previousFrame != None:
                        self.extractPedestriansFromImage(previousFrame, frame, set_name, video_name, i)

                    # TODO save original raw image on disk?
                    #self.save_img(dname, fn, i, frame)

                    previousFrame = frame
                    i += 1


                # after all frames of this video have been preprocessed, we can start creating pairs
                print("Number of created training pairs:", len(self.trainingSamplesPrevious))

                # TODO remove temp code: currently only max 300 pairs for speed up during development
                if len(self.trainingSamplesPrevious) > 300:
                    print("FORCE STOP");
                    break


                print(fn)


        print("Finished frame extraction and matching.")
        print("Calculate additional gradient channels")


        # add gradient channels
        for i in range(len(self.trainingSamplesCurrent)): #indexes 0-4: previous image. 5-9:current image
            self.trainingSamplesPrevious[i] = self.wrapImage(self.trainingSamplesPrevious[i])
            self.trainingSamplesCurrent[i] = self.wrapImage(self.trainingSamplesCurrent[i])

        print("Finished gradient calculations.")

        # convert to np
        self.trainingSamplesPrevious = np.asarray(self.trainingSamplesPrevious, np.float16)
        self.trainingSamplesPrevious = np.swapaxes(self.trainingSamplesPrevious, 1, 3)
        self.trainingSamplesPrevious = np.swapaxes(self.trainingSamplesPrevious, 1, 2)
        self.trainingSamplesCurrent = np.asarray(self.trainingSamplesCurrent, np.float16)
        self.trainingSamplesCurrent = np.swapaxes(self.trainingSamplesCurrent, 1, 3)
        self.trainingSamplesCurrent = np.swapaxes(self.trainingSamplesCurrent, 1, 2)

        if training:
            name = "caltech-training.p"
        else:
            name = "caltech-test.p"

        # saving data to file
        if False: #TODO activate again
            print("saving data to file")
            pickle.dump([self.trainingSamplesPrevious,self.trainingSamplesCurrent], open(os.path.join(self.output_dir, name), "wb"))

        # TODO find better solution for test data fix
        if not training:
            self.testSamplesPrevious = self.trainingSamplesPrevious
            self.testSamplesCurrent = self.trainingSamplesCurrent
            self.trainingSamplesPrevious = None
            self.trainingSamplesCurrent = None

        print("finished")

    def split_into_rgb_channels(self, image):
        '''Split the target image into its red, green and blue channels.
        image - a numpy array of shape (rows, columns, 3).
        output - three numpy arrays of shape (rows, columns) and dtype same as
                 image, containing the corresponding channels.
        '''
        red = image[:, :, 2]
        green = image[:, :, 1]
        blue = image[:, :, 0]
        return red, green, blue

    def wrapImage(self, img):
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sobelx = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=5) #x
        sobely = cv.Sobel(gray_image, cv.CV_16S, 0, 1, ksize=5) #y

        # put everything together
        red, green, blue = self.split_into_rgb_channels(img)
        inputSample = np.array([red, green, blue, sobelx, sobely])

        return inputSample

    def save_img(self, dname, fn, i, frame):
        cv.imwrite('{}/{}_{}_{}.png'.format(
            self.output_dir, os.path.basename(dname),
            os.path.basename(fn).split('.')[0], i), frame)


    def extractPedestriansFromImage(self, img_prev, img_curr, set_name, video_name, frame_i):

        # TODO when saving and loading json file before, we might need frame_i = str(frame_i)

        if frame_i in self.annotations[set_name][video_name]['frames']:
            data = self.annotations[set_name][
                video_name]['frames'][frame_i]
            data_prev = self.annotations[set_name][
                video_name]['frames'][frame_i -1]

            for datum in data:
                ped_id = datum['id']
                x, y, w, h = [int(v) for v in datum['pos']]

                # skip too small images
                if not(w < self.image_size_min_resize or h < self.image_size_min_resize):

                    # go on, only if the pedestrian exists in the previous frame, too
                    ped_exists_in_prev_frame = False
                    for datum_previous in data_prev:
                        if datum_previous['id'] == ped_id:
                            ped_exists_in_prev_frame = True
                            break;

                    if ped_exists_in_prev_frame:

                        img_ped = img_curr[y:y + h, x:x + w]
                        img_ped_prev = img_prev[y:y + h, x:x + w]
                        w_real = len(img_ped[0])
                        h_real = len(img_ped)
                        if not(w_real < self.image_size_min_resize or h_real < self.image_size_min_resize):

                            # resize all images
                            # TODO check how to resize probably for different-ratio image patches
                            resized_image = cv.resize(img_ped, (self.image_width, self.image_height))
                            resized_image_prev = cv.resize(img_ped_prev, (self.image_width, self.image_height))

                            if type(resized_image) != np.ndarray:
                                print("Hugh?")

                            self.trainingSamplesPrevious.append(resized_image_prev)
                            self.trainingSamplesCurrent.append(resized_image)

                            #self.save_img(set_name, video_name, frame_i, img_ped)

                            #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            #n_objects += 1
                        #wri.write(img)


    def getTrainingData(self):

        # TODO call only once
        self.loadDataSet(True)

        # get training data
        XPrevious, XCurrent = self.trainingSamplesPrevious, self.trainingSamplesCurrent
        Yall = self.calcTargetProbMap(XCurrent)

        return XPrevious, XCurrent, Yall

    def calcTargetProbMap(self, currentFrames):

        # calculate target probability maps
        # TODO check generation. right now, all maps are identical as the position of the head has been normalized
        # to a specific position
        # => or should the position still be unnormalized? and only the ratios are normalized? But how can the ped
        # always be at the same position in the previous image??
        sigma = 0.1
        gxyBig = self.gkern(64, sigma)
        leftrightborderremove = (64 - 24) / 2
        gxyFit = gxyBig[:,
                 leftrightborderremove:64 - leftrightborderremove]  # center/center will be used as the normalized position

        # use softmax to allow probability interpretations
        gxyFit = self.softmax(gxyFit)

        # copy same value as target probability map for all training samples
        # TODO if it will really be the same for all, use single representation instead
        Yall = np.full((currentFrames.shape[0], 64, 24), gxyFit)

        return Yall

    def getTestData(self):

        # TODO call only once
        self.loadDataSet(False)

        # TODO get test data
        XPrevious, XCurrent = self.testSamplesPrevious, self.testSamplesCurrent
        Yall = self.calcTargetProbMap(XCurrent)
        return XPrevious, XCurrent , Yall

    # source: http://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


    def softmax(self, allScores):

        # prevent overflow by subtracting a high constant from each term (doesn't change the final result, but the numbers during computation are smaller)
        allScoresSmooth = allScores - allScores.max(axis=1,
                                                    keepdims=True)  # use max per row (not total max) to ensure maximum numerical stability

        # compute all needed values only once
        e = np.exp(allScoresSmooth)

        # keep in mind that allScores[i] contains 10 scores for one image
        # so we need to divide each element by the sum of its row-sum, not by the whole sum
        return e / e.sum(axis=1, keepdims=True)