#import tensorflow as tf
import caltech_loader as cl
import conv_net as cnn
import log
import numpy as np
import sys
import os
import gc

#cf = config
#cfc = user config constant => do not change manually
cf_max_samples = 1000 # maximum number of loaded training samples. 0=unlimited
cfc_cache_dataset_hdd = True # reminder: if this is turned on, and you want to change other settings, they might need a cache reset => clear folder
cf_validation_set_size = 10 #TODO increase! right now we're not yet using the validation data set, that's why it's so tiny # this absolute number of images will be taken from the training images and used as validation data
cf_dataset = 0 # 0 => Caltech
cf_timeout_minutes = 0 # maximum number of minutes used for training. 0=unlimited
cf_min_max_scaling = True #turn on either this or cf_standardization
cf_standardization = True #turn on either this or cf_min_max_scaling
cf_log_auto_save = True #if True, the log file will be saved automatically as soon as all calculations have been finished correctly
cf_log_dir = cf_log_dir_init = "logs"
cf_batch_size = 100 #must divide number of images in all used datasets
cf_learning_rate = 0.9
cf_num_iters = 1000
cf_regularization_strength = 0 #0 means no regularization L2
cf_learning_rate_decay = 0.95 #1 means no decay
cf_dropout_rate = 0.5 # 1.0 = no dropout
cf_optimizer = 2 # 0=GradientDescentOptimizer, 1=AdamOptimizer, 2=MomentumOptimizer(see also cf_momentum)
cf_momentum = 0.9 #0 deactivates the momentum update. when activating/increasing you may want to decrease cf_learning_rate (the other way around, too).

# weight importance of direction vs. distance in accuracy measurements
cf_accuracy_weight_direction = 0.8
cf_accuracy_weight_distance = 1 - cf_accuracy_weight_direction

# relative horizontale position assumed for the pedestrians head in the previous frame.
cf_head_rel_pos_prev_row = 0.25 # 0=top, 1=bottom #TODO verify that the upper and not the lower quarter is used
cf_head_rel_pos_prev_col = 0.5 # 0=left hand side, 1=right hand side, 0.5 = horizontale center


if cf_dataset == 0:
    cfc_dataset_name = 'Caltech'
#else:
#    cfc_dataset_name = 'CIFAR-100'


def redo_finalize(saveLog):

    global XTrainPrevious, XTrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval,\
        cfc_dataset_name, net, cfc_cache_dataset_hdd, cf_min_max_scaling, cf_standardization, cf_max_samples

    # if not already done, load test data
    if (cf_dataset != 0):
        calLoader = cl.CaltechLoader(cfc_datasetpath, cfc_cache_dataset_hdd, cf_max_samples, cf_min_max_scaling, cf_standardization, cf_head_rel_pos_prev_row, cf_head_rel_pos_prev_col)
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



# configurations to automatically evaluate hyperparameters:
param_test_cf_regularization_strength = [
              "cf_regularization_strength", #string name
              [0.25, 0.5, 0.001, 0.002, 0.005, 0.01, 0.1]] #values #0.0001,
param_test_cf_learning_rate = [
    "cf_learning_rate",  # string name
    [0.0001, 0.02, 0.03, 0.04, 0.05, 0.001, 0.01, 0.1, 0.15, 0.2]]  # values
param_test_cf_number_of_filters_conv_both = [
    "cf_number_of_filters_conv_both",  # string name
    [8,16,32,64,128]]  # values
param_test_cf_number_of_filters_conv1 = [
    "cf_number_of_filters_conv1",  # string name
    [8,16,32,64,128]]  # values
param_test_cf_number_of_filters_conv2 = [
    "cf_number_of_filters_conv2",  # string name
    [8, 16, 32, 64,128]]  # values
param_test_cf_number_of_conv_layers = [
    "cf_number_of_conv_layers",  # string name
    [2,1,3,4,5,6]]  # values
param_test_cf_dropout_rate = [
    "cf_dropout_rate",  # string name
    [0.25, 0.75,0.5,1.0]]  # values
param_test_cf_optimizer = [
    "cf_optimizer",  # string name
    [1,0,2]]  # values



no_tests = ["", [0]]
testvals = no_tests
eval_i_max = len(testvals[1])
i=0
while i < eval_i_max: # don't use a for-loop, as we want to manipulate i inside the loop

    #try: #TODO

    # Loading data
    log.log('###############################################################')
    log.log('########################  BEGIN  ##############################')

    if(testvals[0] != ""):

        log.log('DEBUG param ' + testvals[0] + " ({})".format(testvals[1][i]))
        cf_log_dir = os.path.join(cf_log_dir_init, testvals[0]) # save param evals in extra folder

        if testvals[0] == "cf_regularization_strength":
            cf_regularization_strength = testvals[1][i]
        elif testvals[0] == "cf_learning_rate":
            cf_learning_rate = testvals[1][i]
        elif testvals[0] == "cf_number_of_filters_conv_both":
            cf_number_of_filters_conv1 = testvals[1][i]
            cf_number_of_filters_conv2 = testvals[1][i]
        elif testvals[0] == "cf_number_of_filters_conv1":
            cf_number_of_filters_conv1 = testvals[1][i]
        elif testvals[0] == "cf_number_of_filters_conv2":
            cf_number_of_filters_conv2 = testvals[1][i]
        elif testvals[0] == "cf_number_of_conv_layers":
            cf_number_of_conv_layers = testvals[1][i]
        elif testvals[0] == "cf_dropout_rate":
            cf_dropout_rate = testvals[1][i]
        elif testvals[0] == "cf_optimizer":
            cf_optimizer = testvals[1][i]



    if len(sys.argv) == 1:
        # use default file path
        cfc_datasetpath = "/media/th/6C4C-2ECD/ml_datasets"
    else:
        #use user specified file path
        cfc_datasetpath = sys.argv[1]

    log.log('Loading ' + cfc_dataset_name + ' dataset..')
    if(cf_dataset == 0):
        calLoader = cl.CaltechLoader(cfc_datasetpath, cfc_cache_dataset_hdd, cf_max_samples, cf_min_max_scaling, cf_standardization, cf_head_rel_pos_prev_row, cf_head_rel_pos_prev_col)
        XTrainPrevious, XTrainCurrent, Yall = calLoader.getTrainingData()

        # resample training data to gain validation dataset
        # TODO set correct ratio and maybe move to CaltechLoader
        log.log(".. resampling training and validation data")
        indices = np.random.permutation(XTrainPrevious.shape[0])
        trainingset_size = XTrainPrevious.shape[0] - cf_validation_set_size
        train_ids, val_ids = indices[:trainingset_size], indices[trainingset_size:] #kepp in mind: fix params when changing dataset size
        XTrainPrevious, XTrainCurrent, XvalPrevious, XvalCurrent = XTrainPrevious[train_ids, :], XTrainCurrent[train_ids, :], XTrainPrevious[val_ids, :], XTrainCurrent[val_ids, :]
        Ytrain = Yall[train_ids]
        Yval = Yall[val_ids]

        #del Xall
        gc.collect()


    log.log('.. Trainingset includes {} images.'.format(XTrainPrevious.shape[0]))

    log.log('.. Validationset includes {} images.'.format(XvalPrevious.shape[0]))

    # Creating network
    net = cnn.ConvolutionalNetwork(XTrainPrevious, XTrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval,
                                   cf_batch_size,
                                   cf_learning_rate,
                                   cf_num_iters,
                                   cf_regularization_strength,
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
    log.log('.. timeout after {} minutes'.format(cf_timeout_minutes))
    log.log('.. learning rate: {}'.format(cf_learning_rate))
    log.log('.. learning rate decay: {}'.format(cf_learning_rate_decay))
    log.log('.. L2 regularization active: {}'.format(cf_regularization_strength != 0))
    if cf_regularization_strength != 0:
        log.log('.. L2 regularization strength: {}'.format(cf_regularization_strength))
    log.log('.. drop out between S2 and C3 active: {}'.format(cf_dropout_rate > 0 and cf_dropout_rate < 1))
    if cf_dropout_rate > 0 and cf_dropout_rate < 1:
        log.log('.. drop out rate: {}'.format(cf_dropout_rate))
    log.log('.. total number of iterations: {}'.format(cf_num_iters))
    log.log('.. batch size in each iteration: {}'.format(cf_batch_size))
    log.log('.. apply standardization (mean + std): {}'.format(cf_standardization))
    log.log('.. min-max-scaling: {}'.format(cf_min_max_scaling))
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