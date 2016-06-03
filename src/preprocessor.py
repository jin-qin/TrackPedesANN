import tensorflow as tf
import log
import numpy as np
import math
import time
import os

class Preprocessor:

    def __init__(self, data, min_max_scaling=True, standardization=True):

        self.active = min_max_scaling or standardization

        if self.active:
            # preprocessing (will change original data, too!)
            log.log(".. initialize preprocessing")  # train only on previous images
            self.preprocessInit(min_max_scaling, standardization,
                                data)  # call this independently of the values of min_max_scaling or standardization!

            #TODO those learned parameters needs to be saved with the network

            log.log(".. preprocessing initialized")



    # this method needs to be called exactly ONCE before any calls of preprocessData()
    # X must be the training data, no validation data allowed!
    # the calculated values will be reused to preprocess any following data
    # input data X will be changed as if preprocessData(X) has been called, so no further call for this object required
    def preprocessInit(self, min_max_scaling, standardization, Xorig):

        X = Xorig.copy()

        # Min-Max scaling
        if min_max_scaling:
            self.preMin = np.min(X)
            xmax = np.max(X)
            self.preMinMaxScale = xmax - self.preMin

            # we need to apply minmaxscaling once to the temporary data to ensure that the effect for standardization is considered
            if self.preMinMaxScale != 0:
                X -= self.preMin
                X /= self.preMinMaxScale

        else:  # ensure that preprocessData() will not apply min_max_scaling
            self.preMin = 0
            self.preMinMaxScale = 1.0

        # standardization
        if standardization:
            # generate mean image
            self.preMeanImage = np.mean(X, axis=0)

            # normalize by dividing by the standard deviation => unit std
            self.preStd = np.std(X, axis=0)
            self.preStd[self.preStd == 0] = 0.001  # prevent divide by 0

        else:  # ensure that preprocessData() will not apply standardization
            self.preMeanImage = 0
            self.preStd = 1.0

    # apply preprocessing on X
    # preprocessInit needs to be called before
    # will only use params of previously processed training data. so X can be whatever you want, validation data, too.
    def preprocessData(self, Xarr):

        if self.active:

            for X in Xarr:
                # Min-Max scaling
                if self.preMinMaxScale != 0:
                    X -= self.preMin
                    X /= self.preMinMaxScale

                # standardization:
                # subtract mean image from all data => zero mean
                # (this will change the original dataset, too)
                X -= self.preMeanImage

                # normalize by dividing by the standard deviation => unit std
                X /= self.preStd