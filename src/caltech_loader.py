import os
import glob
from scipy.io import loadmat
import cv2 as cv
from collections import defaultdict
import numpy as np
import cPickle as pickle
import log
import preprocessor as pp
import visualizer as vs
import tensorflow as tf

class CaltechLoader:

    def __init__(self, root_dir, cache, maxSamples=5000, min_max_scaling=True, standardization=True,
                 head_rel_pos_prev_row=0.25, head_rel_pos_prev_col=0.5, validation_set_size=1000, image_size_min_resize=48):

        self.trainingSamplesPrevious = None
        self.trainingSamplesCurrent = None
        self.trainingY = None
        self.validationSamplesPrevious = None
        self.validationSamplesCurrent = None
        self.validationY = None
        self.testSamplesPrevious = None
        self.testSamplesCurrent = None
        self.testY = None
        self.annotations = None


        self.image_width = 48
        self.image_height = 128
        self.image_size_min_resize = image_size_min_resize
        self.input_dir = os.path.join(root_dir, "caltech")
        self.output_dir = os.path.join(self.input_dir, "cached")
        self.cache = cache
        self.min_max_scaling = min_max_scaling
        self.standardization = standardization
        self.maxSamples = maxSamples
        self.head_rel_pos_prev_row = head_rel_pos_prev_row
        self.head_rel_pos_prev_col = head_rel_pos_prev_col
        self.validation_set_size = validation_set_size


    def loadDataSet(self, training):

        if training:
            name = "caltech-training.p"
        else:
            name = "caltech-test.p"

        # if no data has been generated yet, generate it once
        cache_file = os.path.join(self.output_dir, name)
        if not self.cache or not os.path.exists(cache_file):
            log.log("No cached dataset has been found. Generate it once.")

            # load annotations, but only once
            if self.annotations is None:
                self.loadAnnotations()

            self.loadImages(training)
        else:
            log.log("Cached caltech dataset has been found. Start loading..")

            log.log("restoring previous preprocessing data")
            self.preprocessor = pickle.load(open(os.path.join(self.output_dir, "caltech-preprocessor.p"), "rb"))

            if training:
                self.trainingSamplesPrevious, self.trainingSamplesCurrent, self.trainingY = pickle.load(open(cache_file, "rb"))
                self.validationSamplesPrevious, self.validationSamplesCurrent, self.validationY = pickle.load(open(os.path.join(self.output_dir, "caltech-validation.p"), "rb"))
            else:
                self.testSamplesPrevious, self.testSamplesCurrent, self.testY = pickle.load(open(cache_file, "rb"))

            log.log("Finished cached data import.")

    # loads all annotations to self.annotations
    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_annotations.py
    def loadAnnotations(self):

        all_obj = 0
        self.annotations = defaultdict(dict)
        for dname in sorted(glob.glob(os.path.join(self.input_dir, 'annotations/set*'))):
            set_name = os.path.basename(dname)
            log.log("Parsing annotations from set {}".format(set_name))
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
                log2 = vbb['A'][0][0][9][0]
                logLen = int(vbb['A'][0][0][10][0][0])

                video_name = os.path.splitext(os.path.basename(anno_fn))[0]
                self.annotations[set_name][video_name]['nFrame'] = nFrame
                self.annotations[set_name][video_name]['maxObj'] = maxObj
                self.annotations[set_name][video_name]['log'] = log2.tolist()
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

        log.log('Total number of annotations: {}'.format(all_obj))

    def get_video_file_names(self, training):

        results = defaultdict(dict)

        imagesets = sorted(glob.glob(os.path.join(self.input_dir, 'set*')))

        image_folders_only = []
        for dname in imagesets:
            if os.path.isdir(dname):
                set_name = os.path.basename(dname)

                if training and self.is_training_set_part(set_name) or (
                    (not training) and self.is_test_set_part(set_name)):
                    image_folders_only.append(dname)

        log.log("{} extracted image set(s) found".format(len(image_folders_only)))
        skipped = len(imagesets) - len(image_folders_only)
        if skipped > 0:
            log.log("{} further file(s) skipped. Forgot to extract?".format(skipped))

        if len(image_folders_only) > 0:

            for dname in image_folders_only:
                set_name = os.path.basename(dname)

                log.log("Loading set {}".format(set_name))

                files = sorted(glob.glob('{}/*.seq'.format(dname)))

                if len(files) > 0:

                    results[set_name] = defaultdict(dict)

                    for fn in files:
                        video_name = os.path.splitext(os.path.basename(fn))[0]
                        results[set_name][video_name] = fn


        return results




    # original source: https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py
    def loadImages(self, training):

        log.log("Loading frames")

        log.log("Min length image for resizing: {}".format(self.image_size_min_resize))

        # create output folder if it doesn't exist yet
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        video_sets = self.get_video_file_names(training)

        x_prev, x_curr, y = [], [], []

        if len(video_sets) > 0:

            for set_name, videos in video_sets.iteritems():
                for video_name, fn in videos.iteritems():

                    #temp, set_name = os.path.split(os.path.split(fn)[0])

                    log.log("Processing video {} of {}".format(video_name, set_name))

                    cap = cv.VideoCapture(fn)
                    previousFrame = None
                    i = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if i > 0:
                            self.extractPedestriansFromImage(previousFrame, frame, set_name, video_name, i, x_prev, x_curr, y)

                        # save original raw image on disk?
                        #self.save_img(dname, fn, i, frame)

                        previousFrame = frame
                        i += 1


                    # after all frames of this video have been preprocessed, we can start creating pairs
                    log.log("Number of created sample pairs: {}".format(len(x_prev)))

                    # allow only max self.maxSamples pairs for speed up (during development)
                    if self.maxSamples > 0 and len(x_prev) > self.maxSamples:
                        log.log("FORCE STOP of dataset loading as the maximum number of samples has been reached");
                        break


            log.log("Finished frame extraction and matching.")
            log.log("Calculate additional gradient channels")


            # add gradient channels
            for i in range(len(x_curr)): #indexes 0-4: previous image. 5-9:current image
                x_prev[i] = self.wrapImage(x_prev[i])
                x_curr[i] = self.wrapImage(x_curr[i])

            log.log("Finished gradient calculations.")

            # convert to np
            x_prev = np.asarray(x_prev, np.float16)
            x_curr = np.asarray(x_curr, np.float16)
            y = np.asarray(y, np.float16)

            # resample training data to gain validation dataset
            # (needs to be done before preprocessing. otherwise validation data will be used for the calculations)
            if training:
                log.log(".. resampling training and validation data")
                indices = np.random.permutation(x_prev.shape[0])
                trainingset_size = x_prev.shape[0] - self.validation_set_size
                train_ids, val_ids = indices[:trainingset_size], indices[trainingset_size:]  # keep in mind: fix params when changing dataset size
                x_prev, x_curr, self.validationSamplesPrevious, self.validationSamplesCurrent = x_prev[train_ids, :], x_curr[train_ids,:],\
                                                                           x_prev[val_ids,:], x_curr[val_ids, :]
                y, self.validationY = y[train_ids], y[val_ids]

                # if we are loading training data, we need to initialize the preprocessor once and also preprocess validation data
                self.preprocessor = pp.Preprocessor(x_prev, self.min_max_scaling, self.standardization)
                self.preprocessor.preprocessData([self.validationSamplesPrevious, self.validationSamplesCurrent])


            # this is valid for training as well as for test data
            self.preprocessor.preprocessData([x_prev, x_curr])


            # (whenever calling other instances (like live camera images in an application) they need to be preprocessed, too)


            # datasets are ready! rename and save..
            if training:

                self.trainingSamplesPrevious = x_prev
                self.trainingSamplesCurrent = x_curr
                self.trainingY = y

                # saving data to file
                if self.cache:
                    name = "caltech-training.p"
                    log.log("saving training data to file")
                    pickle.dump([self.trainingSamplesPrevious, self.trainingSamplesCurrent, self.trainingY],
                                open(os.path.join(self.output_dir, name), "wb"))

                    name = "caltech-validation.p"
                    log.log("saving validation data to file")
                    pickle.dump([self.validationSamplesPrevious, self.validationSamplesCurrent, self.validationY],
                                open(os.path.join(self.output_dir, name), "wb"))

                    name = "caltech-preprocessor.p"
                    log.log("saving preprocessor data to file")
                    pickle.dump(self.preprocessor,
                                open(os.path.join(self.output_dir, name), "wb"))

            else:
                self.testSamplesPrevious = x_prev
                self.testSamplesCurrent = x_curr
                self.testY = y

                # saving data to file
                if self.cache:
                    name = "caltech-test.p"
                    log.log("saving test data to file")
                    pickle.dump([self.testSamplesPrevious, self.testSamplesCurrent, self.testY],
                                open(os.path.join(self.output_dir, name), "wb"))



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

        # convert image to grayscale
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # calc gradients
        # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
        # see http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html#one-important-matter
        sobelx64f = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=5)
        abs_sobelx64f = np.absolute(sobelx64f)
        sobelx_8u = np.uint8(abs_sobelx64f)
        sobely64f = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=5)
        abs_sobely64f = np.absolute(sobely64f)
        sobely_8u = np.uint8(abs_sobely64f)

        # put everything together
        red, green, blue = self.split_into_rgb_channels(img)
        inputSample = np.array([blue, green, red, sobelx_8u, sobely_8u], dtype=np.float16)

        inputSample = np.swapaxes(inputSample, 0, 2)
        inputSample = np.swapaxes(inputSample, 0, 1)

        return inputSample

    def save_img(self, dname, fn, i, frame):
        cv.imwrite('{}/{}_{}_{}.png'.format(
            self.output_dir, os.path.basename(dname),
            os.path.basename(fn).split('.')[0], i), frame)

    def extractPedestriansFromImage(self, img_prev, img_curr, set_name, video_name, frame_i, x_prevArr, x_currArr, y_labels):

        # keep in mind: when saving and loading json file before, we might need frame_i = str(frame_i)
        try:
            in_prev_frame = (frame_i -1) in self.annotations[set_name][video_name]['frames']
            in_curr_frame = frame_i in self.annotations[set_name][video_name]['frames']

        except Exception as e:
            in_curr_frame = in_prev_frame = False

            log,log("Error during annotation check. Have you forgotten to provide annotations for all datasets?")



        if in_prev_frame and in_curr_frame:

            data = self.annotations[set_name][video_name]['frames'][frame_i]
            data_prev = self.annotations[set_name][video_name]['frames'][(frame_i -1)]

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
                            x_prev, y_prev, w_prev, h_prev = [int(v) for v in datum_previous['pos']]
                            break;

                    if ped_exists_in_prev_frame:

                        # pedestrians face is supposed to be at the center of the given rectangle
                        img_ped = img_curr[y:y + h, x:x + w]
                        img_ped_prev = img_prev[y:y + h, x:x + w]


                        w_real = len(img_ped[0])
                        h_real = len(img_ped)
                        if not(w_real < self.image_size_min_resize or h_real < self.image_size_min_resize):

                            # resize all images
                            resized_image = cv.resize(img_ped, (self.image_width, self.image_height))

                            resized_image_prev = cv.resize(img_ped_prev, (self.image_width, self.image_height))
                            w_resize_scale = self.image_width / float(w_real)
                            h_resize_scale = self.image_height / float(h_real)
                            if type(resized_image) != np.ndarray:
                                log.log("Hugh?")

                            x_prevArr.append(resized_image_prev)
                            x_currArr.append(resized_image)

                            # potential head position is supposed to be in the center of the current frame
                            # but as we choose the image patch based on the previous and not the current frame,
                            # the relative position is not guaranteed to be in the center. Calculate the relative
                            # position and use it to generate a target probability map.
                            absolute_center_x = x + (w * self.head_rel_pos_prev_col)
                            absolute_center_y = y + (h * self.head_rel_pos_prev_row) #not center, but upper quarter
                            relative_center_x = absolute_center_x - x_prev
                            relative_center_y = absolute_center_y - y_prev

                            # scale center by same ratio as image
                            relative_center_x *= w_resize_scale
                            relative_center_y *= h_resize_scale

                            # and halve again, because the output probability map has only 2px accuracy
                            relative_center_x /= 2
                            relative_center_y /= 2

                            relative_center_x = int(round(relative_center_x))
                            relative_center_y = int(round(relative_center_y))

                            probs = self.calcTargetProbMap(relative_center_x, relative_center_y)
                            y_labels.append( probs )

                            # you can visualize this if you want:
                            if False:
                                img_test = np.full([64, 24, 3], 255, np.uint8)
                                img_test[relative_center_y][relative_center_x] = (255, 0, 0)
                                vis = vs.Visualizer()
                                cv.imshow('ped prev', img_ped_prev)
                                cv.imshow('frame current', img_curr)
                                cv.imshow('ped current', img_ped)
                                #cv.imshow('probability map', probs)
                                # cv.imshow('probability map', img_test)
                                vis.visualizeProbabilityMap(probs)

                            #self.save_img(set_name, video_name, frame_i, img_ped)

                            #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            #n_objects += 1
                        #wri.write(img)


    def getTrainingData(self):

        # call only once
        if self.trainingSamplesPrevious is None:
            self.loadDataSet(True)

        return self.trainingSamplesPrevious, self.trainingSamplesCurrent, self.trainingY


    def getValidationData(self):

        # call only once
        if self.trainingSamplesPrevious is None: #validation data will be loaded together with validation data
            self.loadDataSet(True)

        return self.validationSamplesPrevious, self.validationSamplesCurrent, self.validationY


    def calcTargetProbMap(self, center_x, center_y):

        # calculate target probability maps
        sigma = 1
        sigma_square = sigma * sigma
        scores = np.zeros([64,24])
        for x in range(24):
            for y in range(64):
                scores[y][x] = np.exp(-( np.square(x - center_x) + np.square(y - center_y)) / (2 * sigma_square))

        # use softmax to allow probability interpretations
        # Update: we do this in tensorflow
        #probs = self.softmax(scores)

        return scores

    def getTestData(self):

        # call only once
        if self.testSamplesPrevious is None:
            self.loadDataSet(False)

        return self.testSamplesPrevious, self.testSamplesCurrent, self.testY

    def softmax(self, allScores):

        # prevent overflow by subtracting a high constant from each term (doesn't change the final result, but the numbers during computation are smaller)
        allScoresSmooth = allScores - allScores.max(axis=1,
                                                    keepdims=True)  # use max per row (not total max) to ensure maximum numerical stability

        # compute all needed values only once
        e = np.exp(allScoresSmooth)

        # keep in mind that allScores[i] contains 10 scores for one image
        # so we need to divide each element by the sum of its row-sum, not by the whole sum
        return e / e.sum(axis=1, keepdims=True)


    # returns true, if folder_name is the name of a folder containing TRAINING data
    def is_training_set_part(self, folder_name):
        return folder_name == "set00" or folder_name == "set01" or folder_name == "set02" or folder_name == "set03" or folder_name == "set04" or folder_name == "set05"

    # returns true, if folder_name is the name of a folder containing TEST data
    def is_test_set_part(self, folder_name):
        return folder_name == "set06" or folder_name == "set07" or folder_name == "set08" or folder_name == "set09" or folder_name == "set10"

    def is_caltech_dataset_folder(self, folder_name):
        return self.is_training_set_part(folder_name) or self.is_test_set_part(folder_name)

    def get_preprocessor(self):
        return self.preprocessor

    def get_annotations(self):

        if self.annotations is None:
            self.loadAnnotations()

        return self.annotations