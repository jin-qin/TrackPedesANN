'''
This network is based on the paper "Human Tracking using Convolutional Neural Networks" by J Fan - 2010

Installation Requirements:
- python 2 (python 3 not yet supported!)
- opencv 2.4 (opencv 3 not yet supported!)
- tensorflow (currently version 0.8)
- numpy

How to run:
- Download the Caltech Dataset from http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/
- Extract the different sets into one common folder
- Set the var cfc_datasetpath_init (see below) to fit that folder
- If you don't got enough memory to extract or process all data at once, you need at least:
    + annotations
    + one extracted archive containing training data (set00 - set05)
    + one extracted archive containing test data (set06 - set10)
'''

###############################################################
############# CONFIGURATION to play around ####################
###############################################################

# general
cf_max_samples = 300 # maximum number of loaded ([training + validation] or test) samples. 0=unlimited
cf_num_iters = 450
cf_batch_size = 50
cf_validation_set_size = round(cf_max_samples * 0.1) # this absolute number of images will be taken from the training images and used as validation data
cf_min_max_scaling = True #turn on either this or cf_standardization
cf_standardization = True #turn on either this or cf_min_max_scaling
cf_learning_rate = 0.9
cf_learning_rate_decay = 0.95 #1 means no decay
cf_dropout_rate = 0.5 # 1.0 = no dropout
cf_optimizer = 2 # 0=GradientDescentOptimizer, 1=AdamOptimizer, 2=MomentumOptimizer(see also cf_momentum)
cf_momentum = 0.9 #0 deactivates the momentum update. when activating/increasing you may want to decrease cf_learning_rate (the other way around, too).
cf_accuracy_weight_direction = 0.8 # weight importance of direction vs. distance in accuracy measurements (distance = 1 - direc)

# make your life convenient
cfc_cache_dataset_hdd = True # reminder: if this is turned on, and you want to change other settings, they might need a cache reset => clear folder
cf_timeout_minutes = 0 # maximum number of minutes used for training. 0=unlimited
cf_log_auto_save = True #if True, the log file will be saved automatically as soon as all calculations have been finished correctly
cf_log_dir = cf_log_dir_init = "logs"
cfc_datasetpath_init = "/media/th/6C4C-2ECD/ml_datasets" # path to the (Caltech) dataset. Can be overriden by commandline parameter.

# relative horizontale position assumed for the pedestrians head in the previous frame.
cf_head_rel_pos_prev_row = 0.25 # 0=top, 1=bottom
cf_head_rel_pos_prev_col = 0.5 # 0=left hand side, 1=right hand side, 0.5 = horizontale center

# extracted image patches will be resized to 48x128px,
# but we will only resize images with at least one dimension having a minimum of:
cf_image_size_min_resize = 48 * 1.25


###############################################################
### STATIC CONFIG - Please don't change anything ##############
###############################################################

# 0 => Caltech
cf_dataset = 0

cf_accuracy_weight_distance = 1 - cf_accuracy_weight_direction

# validation set needs at least as much images as included in one batch in order to allow accuracy evaluation
cf_validation_set_size = max(cf_validation_set_size, cf_batch_size)

if cf_dataset == 0:
    cfc_dataset_name = 'Caltech'
#else:
#    cfc_dataset_name = 'CIFAR-100'


###############################################################
###################### Imports ################################
###############################################################

import caltech_loader as cl
import conv_net as cnn
import log
import numpy as np
import sys
import os
import gc


###############################################################
############# Parameter Tuning Part 1  ### ####################
###############################################################

# configurations to automatically evaluate hyperparameters:
param_test_cf_learning_rate = [
    "cf_learning_rate",  # string name
    [0.001, 0.01, 0.1, 0.2,0.5,0.9,1,1.5]]  # values
param_test_cf_batch_size = [
    "cf_batch_size",  # string name
    [50,100,150,200,250]]  # values
param_test_cf_learning_rate_decay = [
    "cf_learning_rate_decay",  # string name
    [0.5, 0.7, 0.9, 0.95, 0.99, 1]]  # values
param_test_cf_momentum = [
    "cf_momentum",  # string name
    [0, 0.25, 0.5, 0.72, 1]]  # values
param_test_cf_dropout_rate = [
    "cf_dropout_rate",  # string name
    [0.25, 0.75,0.5,1.0]]  # values
param_test_cf_optimizer = [
    "cf_optimizer",  # string name
    [1,0,2]]  # values
param_test_cf_image_size_min_resize = [
    "cf_image_size_min_resize",  # string name
    [48 * 0.5, 48 * 0.75, 48, 48 * 1.2, 48 * 1.5, 48 * 2]]  # values
param_test_cf_image_size_min_resize = [
    "cf_min_max_scaling",  # TODO CLEAN CACHE AFTER EACH ITERATION (of this parameter)!!
    [True, False]]  # values
param_test_cf_standardization = [
    "cf_standardization",  # TODO CLEAN CACHE AFTER EACH ITERATION (of this parameter)!!
    [True, False]]  # values



no_tests = ["", [0]]
######################## set testvals to the var that should be tuned, or = no_tests to turn off tuning!
testvals = no_tests


###############################################################
################### After Training  ###########################
###############################################################

# this function will be called AFTER training. Or as soon as the user is aborting training.
def redo_finalize(saveLog):

    global XTrainPrevious, XTrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval, calLoader,\
        cfc_dataset_name, net, cfc_cache_dataset_hdd, cf_min_max_scaling, cf_standardization, cf_max_samples, cf_validation_set_size

    # if not already done, load test data
    log.log("No testset available yet, start loading it..")
    XtestPrev, XtestCurr, Ytest = calLoader.getTestData()
    log.log('Loaded testset, which includes {} images.'.format(XtestPrev.shape[0]))

    val_acc = net.accuracy(XvalPrevious, XvalCurrent, Yval)
    log.log('FINAL Accuracy validation {0:.3f}%'.format(val_acc * 100))

    test_acc = net.accuracy(XtestPrev, XtestCurr, Ytest)
    log.log('FINAL Accuracy test {0:.3f}%'.format(test_acc * 100))

    train_acc = net.accuracy(XTrainPrevious, XTrainCurrent, Ytrain)
    log.log('FINAL Accuracy training {0:.3f}%'.format(train_acc * 100))

    net.closeSession()
    log.log('.. training finished.')


    runtime = net.getTrainingRuntimeSeconds()
    minutes = np.floor(runtime / 60)
    seconds = round(((runtime / 60) - minutes) * 60, 2)
    runtime_text = "{} minutes and {} seconds".format(minutes, seconds)
    log.log("total runtime used for training: " + runtime_text)

    log.log('########################  END  ################################')
    log.log('###############################################################')

    if saveLog:
        log.logSetName(cfc_dataset_name + '-{}p'.format(round(test_acc * 100, 2)))
        log.logSave(cf_log_dir)




###############################################################
############### Main Program: Training  #######################
###############################################################

eval_i_max = len(testvals[1])
i=0
while i < eval_i_max: # don't use a for-loop, as we want to manipulate i inside the loop

    #try: #TODO

    # Loading data
    log.log('###############################################################')
    log.log('########################  BEGIN  ##############################')


    ############# Parameter Tuning Part 2  ### ###################
    if(testvals[0] != ""):

        log.log('DEBUG param ' + testvals[0] + " ({})".format(testvals[1][i]))
        cf_log_dir = os.path.join(cf_log_dir_init, testvals[0]) # save param evals in extra folder

        if testvals[0] == "cf_learning_rate":
            cf_learning_rate = testvals[1][i]
        elif testvals[0] == "cf_batch_size":
            cf_batch_size = testvals[1][i]
        elif testvals[0] == "cf_learning_rate_decay":
            cf_learning_rate_decay = testvals[1][i]
        elif testvals[0] == "cf_momentum":
            cf_momentum = testvals[1][i]
            cf_optimizer = 2
        elif testvals[0] == "cf_image_size_min_resize":
            cf_image_size_min_resize = testvals[1][i]
        elif testvals[0] == "cf_dropout_rate":
            cf_dropout_rate = testvals[1][i]
        elif testvals[0] == "cf_optimizer":
            cf_optimizer = testvals[1][i]
        elif testvals[0] == "cf_min_max_scaling":
            cf_min_max_scaling = testvals[1][i]
        elif testvals[0] == "cf_standardization":
            cf_standardization = testvals[1][i]


    if len(sys.argv) == 1:
        # use default file path
        cfc_datasetpath = cfc_datasetpath_init
    else:
        #use user specified file path
        cfc_datasetpath = sys.argv[1]

    # load data
    log.log('Loading ' + cfc_dataset_name + ' dataset..')
    if(cf_dataset == 0):
        calLoader = cl.CaltechLoader(cfc_datasetpath, cfc_cache_dataset_hdd, cf_max_samples, cf_min_max_scaling,
                                     cf_standardization, cf_head_rel_pos_prev_row, cf_head_rel_pos_prev_col,
                                     cf_validation_set_size, cf_image_size_min_resize)
        XTrainPrevious, XTrainCurrent, Ytrain = calLoader.getTrainingData()
        XvalPrevious, XvalCurrent, Yval = calLoader.getValidationData()

        gc.collect()

    log.log('.. Trainingset includes {} images.'.format(XTrainPrevious.shape[0]))
    log.log('.. Validationset includes {} images.'.format(XvalPrevious.shape[0]))


    # Creating network
    net = cnn.ConvolutionalNetwork(XTrainPrevious, XTrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval,
                                   cf_batch_size,
                                   cf_learning_rate,
                                   cf_num_iters,
                                   cf_learning_rate_decay,
                                   cf_momentum,
                                   cf_timeout_minutes,
                                   cf_log_dir,
                                   cf_dropout_rate,
                                   cf_optimizer,
                                   cf_head_rel_pos_prev_row,
                                   cf_head_rel_pos_prev_col,
                                   cf_accuracy_weight_direction,
                                   cf_accuracy_weight_distance)

    # Training
    log.log('Start Training..')
    if cf_timeout_minutes  > 0:
        log.log('.. timeout after {} minutes'.format(cf_timeout_minutes))
    log.log('.. total number of iterations: {}'.format(cf_num_iters))
    log.log('.. batch size in each iteration: {}'.format(cf_batch_size))
    log.log('.. learning rate: {}'.format(cf_learning_rate))
    log.log('.. learning rate decay: {}'.format(cf_learning_rate_decay))
    log.log('.. momentum update: {}'.format(cf_momentum))


    try:
        net.train()
    except KeyboardInterrupt:
        log.log("WARNING: User interrupted progess. Saving latest results.")

        finalizeAndSave = input("Do you want to save the latest data? [y/n]")
        if finalizeAndSave != "n":
            log.log("Saving latest results.")
            redo_finalize(True)
            sys.exit()
        else:
            log.log("Results deleted.")

    redo_finalize(cf_log_auto_save)

    #except Exception as e:  #TODO
    #    log.log("crash detected. auto repairing.. redo.. " + e.message)
    #    i -= 1 # on error: redo this iteration

    i += 1